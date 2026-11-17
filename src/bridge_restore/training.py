"""Single-device training. Dataset and degradation internals are not inspected."""
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import json
import time
import copy
from tqdm.auto import tqdm
from .evaluation.selection import BestSelection
from dataclasses import replace
from .evaluation import RestorationEvaluator
from .sampling import SDESampler
import torch
from .checkpoint import capture_rng_state, restore_rng_state, save_checkpoint, load_checkpoint


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.averaged_parameters = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            target = self.averaged_parameters[key]
            if target.is_floating_point():
                target.lerp_(value.detach(), 1 - self.decay)
            else:
                target.copy_(value)

    def copy_to(self, model):
        model.load_state_dict(self.averaged_parameters)

    def state_dict(self):
        return {"decay": self.decay, "parameters": self.averaged_parameters}

    def load_state_dict(self, state):
        if state["decay"] != self.decay:
            raise ValueError("EMA decay mismatch")
        for key, value in state["parameters"].items():
            self.averaged_parameters[key].copy_(value)


class Trainer:
    """step counts successful optimizer updates. Accumulation weights by sample count.

    Validation reports raw-model weighted denoising loss using fixed noise/time
    draws; it is not a restoration metric. Test data are never passed here.
    Exact data-order resume uses EpochSampler and deterministic datasets.
    """
    def __init__(self, model, bridge, train_loader, val_loader, config,
                 split_checksum, corruption_description, data_bundle=None):
        self.config = config
        self.device = torch.device(config.training.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use training.device: cpu")
        if config.training.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested CUDA bfloat16 is unsupported")
        self.model = model.to(self.device)
        self.bridge = bridge
        self.train_loader = train_loader
        self.val_loader = val_loader
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError("Training and validation loaders must be nonempty")
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr,
                                           weight_decay=config.training.weight_decay)
        self.ema = EMA(model, config.training.ema_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.training.precision == "float16")
        self.step = 0
        self.data_position = {"epoch": 0, "batch": 0}
        self._iterator = None
        self.split_checksum = split_checksum
        self.corruption_description = corruption_description
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_value = None
        self.selection = BestSelection(config.evaluation.resolved_criteria())
        self.last_validation_step = -1
        self.samples_seen = 0
        self.evaluator = None
        self.data_bundle = data_bundle
        if config.evaluation.enabled:
            if data_bundle is None: raise ValueError("Evaluation requires DataBundle")
            self.evaluation_model = copy.deepcopy(self.model).eval().requires_grad_(False) if config.evaluation.weights == "ema" else self.model
            self.evaluator = RestorationEvaluator(val_loader.dataset, data_bundle.corruption,
                SDESampler(bridge, **asdict(config.sampling)), config.evaluation, self.device)


    def _autocast(self):
        precision = self.config.training.precision
        if precision == "float32":
            return nullcontext()
        dtype = torch.float16 if precision == "float16" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _next_batch(self):
        if self._iterator is None:
            # Rebuilding/skipping loader must not advance training noise/dropout RNG.
            rng = capture_rng_state()
            sampler = self.train_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.data_position["epoch"])
            if self.train_loader.generator is not None:
                self.train_loader.generator.manual_seed(self.config.seed + self.data_position["epoch"])
            self._iterator = iter(self.train_loader)
            for _ in range(self.data_position["batch"]):
                next(self._iterator)
            restore_rng_state(rng)
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.data_position = {"epoch": self.data_position["epoch"] + 1, "batch": 0}
            self._iterator = None
            return self._next_batch()
        self.data_position["batch"] += 1
        return batch

    def train_step(self):
        self.model.train()
        remaining = len(self.train_loader) - self.data_position["batch"]
        if remaining == 0: remaining = len(self.train_loader)
        batches = [self._next_batch() for _ in range(min(self.config.training.grad_accum_steps, remaining))]
        count = sum(batch["x0"].shape[0] for batch in batches)
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.
        for batch in batches:
            x0 = batch["x0"].to(self.device, non_blocking=True)
            xT = batch["xT"].to(self.device, non_blocking=True)
            with self._autocast():
                losses = self.bridge.training_loss(self.model, x0, xT)
                loss = losses.sum() / count
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            self.scaler.scale(loss).backward()
            total += loss.detach().item()
        self.scaler.unscale_(self.optimizer)
        if not self.scaler.is_enabled() and any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise FloatingPointError("Nonfinite gradients")
        previous_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        updated = self.scaler.get_scale() >= previous_scale
        if updated:
            self.step += 1
            self.ema.update(self.model)
            self.samples_seen += count
        return {"loss": total, "batch_samples": count, "updated": updated,
                "lr": self.optimizer.param_groups[0]["lr"]}

    @torch.no_grad()
    def validate(self):
        was_training = self.model.training
        self.model.eval()
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed + 100000)
        total, count = 0., 0
        try:
            for index, batch in enumerate(tqdm(self.val_loader, desc="Validation loss", unit="batch", leave=False)):
                if index >= self.config.training.validation_batches:
                    break
                with self._autocast():
                    losses = self.bridge.training_loss(self.model, batch["x0"].to(self.device),
                                                       batch["xT"].to(self.device), generator)
                total += losses.sum().item()
                count += len(losses)
            if not torch.isfinite(torch.tensor(total)):
                raise FloatingPointError("Nonfinite validation loss")
            return {"validation_loss": total / count, "validation_samples": count}
        finally:
            self.model.train(was_training)

    def _log(self, values):
        record = {"step": self.step, **values}
        with (self.output_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        tqdm.write(json.dumps(record))

    @property
    def completed_epochs(self):
        return self.data_position["epoch"] + int(self.data_position["batch"] == len(self.train_loader))

    def fit(self):
        cfg = self.config.training
        start = time.monotonic()
        skipped = 0
        def active():
            return ((cfg.max_steps is None or self.step < cfg.max_steps) and
                    (cfg.max_epochs is None or self.completed_epochs < cfg.max_epochs))
        while active():
            # train_step advances the epoch only when fetching its first batch.
            at_end = self.data_position["batch"] == len(self.train_loader)
            epoch = self.data_position["epoch"] + 1 + int(at_end)
            initial = 0 if at_end else self.data_position["batch"]
            with tqdm(total=len(self.train_loader), initial=initial,
                      desc=f"Train epoch {epoch}", unit="batch") as progress:
                while active():
                    previous = 0 if self.data_position["batch"] == len(self.train_loader) else self.data_position["batch"]
                    metrics = self.train_step()
                    progress.update(self.data_position["batch"] - previous)
                    progress.set_postfix(loss=f"{metrics['loss']:.4f}", step=self.step)
                    if not metrics["updated"]:
                        skipped += 1
                        if skipped >= 20:
                            raise FloatingPointError("20 consecutive AMP overflows")
                    else:
                        skipped = 0
                        if self.step == 1 or self.step % cfg.log_interval == 0:
                            self._log({**metrics, "epoch": epoch, "elapsed_seconds": time.monotonic() - start})
                    end_epoch = self.data_position["batch"] == len(self.train_loader)
                    if self.evaluator is not None:
                        ec = self.config.evaluation
                        due = (metrics["updated"] and self.step % ec.interval == 0 if ec.unit == "step"
                               else end_epoch and epoch % ec.interval == 0)
                        if due and self.last_validation_step != self.step:
                            self.evaluate_restoration()
                    elif metrics["updated"] and self.step % cfg.validation_interval == 0:
                        self._log(self.validate())
                    if end_epoch:
                        break
        if self.evaluator is not None:
            if self.last_validation_step != self.step:
                self.evaluate_restoration()
            self.test_best()
        else:
            # Legacy loss-only training has no restoration selection criterion.
            self.save(self.output_dir / "checkpoints" / "last.pt")

    def evaluate_restoration(self):
        ec = self.config.evaluation
        if ec.weights == "ema": self.ema.copy_to(self.evaluation_model)
        report = self.evaluator.evaluate(self.evaluation_model)
        report.protocol.update({"step": self.step, "epoch": self.completed_epochs,
                                "split": "val", "split_checksum": self.split_checksum,
                                "corruption": self.corruption_description, "experiment": asdict(self.config)})
        directory = (f"epoch_{self.completed_epochs:04d}" if self.data_position["batch"] == len(self.train_loader)
                     else f"step_{self.step:08d}")
        report.save(self.output_dir / "validation" / directory)
        self.last_validation_step = self.step
        self._log({"validation": report.summary, "completed_epochs": self.completed_epochs,
                   "samples_seen": self.samples_seen})
        improved = self.selection.update(report.summary, self.step, self.completed_epochs)
        # All criteria have been updated before any checkpoint serializes them.
        for name in improved:
            path = self.output_dir / "best" / name
            report.protocol["criterion"] = name
            report.save(path)
            self.save(path / "checkpoint.pt")

    def test_best(self):
        ec = self.config.evaluation
        selected = [name for name, rule in self.selection.criteria.items() if rule["test"]]
        if not selected:
            return
        test_config = replace(ec, visual_samples_path=ec.test_visual_samples_path,
            subset_size=ec.test_subset_size, metrics=ec.test_metrics or ec.metrics,
            criteria={}, selection_metric=next(iter(ec.test_metrics or ec.metrics)))
        # Release validation metric networks before loading test metrics.
        self.evaluator = None
        evaluator = RestorationEvaluator(self.data_bundle.test_loader.dataset, self.data_bundle.corruption,
            SDESampler(self.bridge, **asdict(self.config.sampling)), test_config, self.device, split="test")
        model = self.evaluation_model if ec.weights == "ema" else copy.deepcopy(self.model)
        tested = {}
        for name in selected:
            checkpoint = self.output_dir / "best" / name / "checkpoint.pt"
            state = load_checkpoint(checkpoint)
            key = (state["step"], ec.weights)
            if key in tested:
                report = tested[key]
            else:
                model.load_state_dict(state["ema"]["parameters"] if ec.weights == "ema" else state["model"])
                report = evaluator.evaluate(model)
                tested[key] = report
            report.protocol.update({"checkpoint_step": state["step"], "epoch": state["data_position"]["epoch"] +
                int(state["data_position"]["batch"] == len(self.train_loader)), "checkpoint": str(checkpoint),
                "criterion": name, "split": "test", "split_checksum": self.split_checksum})
            report.save(self.output_dir / "test" / name)

    def save(self, path):
        save_checkpoint(path, {
            "format_version": 1, "model": self.model.state_dict(), "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
            "step": self.step, "data_position": self.data_position,
            "best_value": self.best_value, "best_by_criterion": self.selection.best, "last_validation_step": self.last_validation_step, "samples_seen": self.samples_seen,
            "config": asdict(self.config), "split_checksum": self.split_checksum,
            "corruption_description": self.corruption_description, "rng": capture_rng_state()})

    @staticmethod
    def save_resume_copy(path, state):
        save_checkpoint(path, state)

    def resume(self, path):
        state = load_checkpoint(path)
        old, new = state["config"], asdict(self.config)
        if asdict(type(self.config.evaluation)(**old.get("evaluation", new["evaluation"]))) != new["evaluation"]: raise ValueError("Resume evaluation configuration mismatch")
        self.best_value = state.get("best_value")
        self.selection.best = state.get("best_by_criterion", {})
        self.last_validation_step = state.get("last_validation_step", -1)
        self.samples_seen = state.get("samples_seen", 0)
        if self.evaluator is not None and not self.selection.best:
            self.last_validation_step = -1  # legacy checkpoint: select under the new layout
        # Only operational changes and sampling settings may change on resume.
        for key in ("seed", "model", "bridge", "corruption"):
            if old[key] != new[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        old_data, new_data = dict(old["data"]), dict(new["data"])
        old_data.pop("num_workers"); new_data.pop("num_workers")
        if old_data != new_data:
            raise ValueError("Resume data configuration mismatch")
        for key in ("batch_size", "grad_accum_steps", "lr", "weight_decay", "precision", "ema_decay"):
            if old["training"][key] != new["training"][key]:
                raise ValueError(f"Resume training configuration mismatch: {key}")
        if state["split_checksum"] != self.split_checksum or state["corruption_description"] != self.corruption_description:
            raise ValueError("Resume split/corruption mismatch")
        # Other criteria may have improved after this historical checkpoint.
        # Their files are not part of this state: never silently mix branches.
        for name, record in self.selection.best.items():
            path = self.output_dir / "best" / name / "checkpoint.pt"
            if path.exists():
                saved = load_checkpoint(path)
                if saved["step"] != record["step"]:
                    raise ValueError("Resume best-checkpoint history mismatch; start a new output directory")
            else:
                if record["step"] == state["step"]:
                    self.save_resume_copy(path, state)
                else:
                    raise ValueError(f"Missing historical best checkpoint for {name}; copy its best directory before resume")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.ema.load_state_dict(state["ema"])
        self.scaler.load_state_dict(state["scaler"])
        self.step = state["step"]
        self.data_position = dict(state["data_position"])
        self._iterator = None
        restore_rng_state(state["rng"])
