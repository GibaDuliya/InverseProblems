"""Single-device training. Dataset and degradation internals are not inspected."""
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import json
import time
import copy
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
            for index, batch in enumerate(self.val_loader):
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
        print(json.dumps(record), flush=True)

    def fit(self):
        cfg = self.config.training
        start = time.monotonic()
        skipped = 0
        while self.step < cfg.max_steps:
            metrics = self.train_step()
            if not metrics["updated"]:
                skipped += 1
                if skipped >= 20:
                    raise FloatingPointError("20 consecutive AMP overflows; inspect precision and scales")
                continue
            skipped = 0
            if self.step == 1 or self.step % cfg.log_interval == 0:
                self._log({**metrics, "elapsed_seconds": time.monotonic() - start})
            if self.evaluator is not None:
                ec = self.config.evaluation
                epoch = self.data_position["epoch"] + 1
                due = (self.step % ec.interval == 0 if ec.unit == "step" else
                       self.data_position["batch"] == len(self.train_loader) and epoch % ec.interval == 0)
                if due: self.evaluate_restoration()
            elif self.step % cfg.validation_interval == 0:
                self._log(self.validate())
            if self.step % cfg.checkpoint_interval == 0:
                self.save(self.output_dir / "checkpoints" / f"step_{self.step:08d}.pt")
        if self.evaluator is not None and self.last_validation_step != self.step:
            self.evaluate_restoration()
        self.save(self.output_dir / "checkpoints" / "last.pt")
        if self.evaluator is not None and self.config.evaluation.final_test:
            state = load_checkpoint(self.output_dir / "checkpoints" / "best.pt")
            ec = self.config.evaluation
            model = self.evaluation_model if ec.weights == "ema" else copy.deepcopy(self.model)
            model.load_state_dict(state["ema"]["parameters"] if ec.weights == "ema" else state["model"])
            test_config = replace(ec, visual_samples_path=ec.test_visual_samples_path, subset_size=ec.test_subset_size, metrics=ec.test_metrics or ec.metrics,
                                  selection_metric=next(iter(ec.test_metrics or ec.metrics)))
            evaluator = RestorationEvaluator(self.data_bundle.test_loader.dataset, self.data_bundle.corruption,
                SDESampler(self.bridge, **asdict(self.config.sampling)), test_config, self.device, split="test")
            report = evaluator.evaluate(model)
            report.protocol.update({"checkpoint_step": state["step"], "split": "test", "split_checksum": self.split_checksum})
            report.save(self.output_dir / "test")

    def evaluate_restoration(self):
        ec = self.config.evaluation
        if ec.weights == "ema": self.ema.copy_to(self.evaluation_model)
        report = self.evaluator.evaluate(self.evaluation_model)
        report.protocol.update({"step": self.step, "split": "val", "split_checksum": self.split_checksum,
                                "corruption": self.corruption_description, "experiment": asdict(self.config)})
        report.save(self.output_dir / "validation" / f"step_{self.step:08d}")
        self.last_validation_step = self.step
        value = report.summary["restoration"][ec.selection_metric]["mean"]
        if value != value: raise FloatingPointError("NaN selection metric")
        self._log({"validation": report.summary, "completed_epochs": self.data_position["epoch"] +
                   int(self.data_position["batch"] == len(self.train_loader)), "samples_seen": self.samples_seen})
        if self.best_value is None or (value < self.best_value if ec.selection_mode == "min" else value > self.best_value):
            self.best_value = value
            self.save(self.output_dir / "checkpoints" / "best.pt")

    def save(self, path):
        save_checkpoint(path, {
            "format_version": 1, "model": self.model.state_dict(), "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
            "step": self.step, "data_position": self.data_position,
            "best_value": self.best_value, "last_validation_step": self.last_validation_step, "samples_seen": self.samples_seen,
            "config": asdict(self.config), "split_checksum": self.split_checksum,
            "corruption_description": self.corruption_description, "rng": capture_rng_state()})

    def resume(self, path):
        state = load_checkpoint(path)
        old, new = state["config"], asdict(self.config)
        if old.get("evaluation", new["evaluation"]) != new["evaluation"]: raise ValueError("Resume evaluation configuration mismatch")
        self.best_value = state.get("best_value")
        self.last_validation_step = state.get("last_validation_step", -1)
        self.samples_seen = state.get("samples_seen", 0)
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
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.ema.load_state_dict(state["ema"])
        self.scaler.load_state_dict(state["scaler"])
        self.step = state["step"]
        self.data_position = dict(state["data_position"])
        self._iterator = None
        restore_rng_state(state["rng"])
