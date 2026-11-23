"""Train an on-policy SOC student and select it by validation reward."""
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import torch

from bridge_restore.checkpoint import capture_rng_state, restore_rng_state, save_checkpoint
from bridge_restore.evaluation.report import safe

from .adjoint_matching import adjoint_matching_loss
from .checkpoint import load_soc_checkpoint
from .evaluation import SOCEvaluator, evaluation_ids, to_device


class SOCTrainer:
    def __init__(self, teacher, student, sampler, reward, data, config, provenance):
        self.teacher = teacher.requires_grad_(False).eval()
        self.student = student.eval()
        self.sampler, self.reward, self.data = sampler, reward, data
        self.config, self.provenance = config, provenance
        self.output = Path(config.output_dir)
        self.optimizer = torch.optim.Adam(student.parameters(), lr=config.training.lr)
        self.generator = torch.Generator(device=config.training.device).manual_seed(config.seed)
        self.step = self.samples_seen = 0
        self.data_position = (0, 0)
        self.last_validation_step = None
        self.best = None
        self.evaluator = None
        self.resumed = False
        self.migrate_validation = False
        self.evaluation_samples = {
            split: evaluation_ids(loader.dataset, config, split)
            for split, loader in (("val", data.val_loader), ("test", data.test_loader))
        }

    def train_step(self, batches):
        sizes = [len(batch["xT"]) for batch in batches]
        samples = sum(sizes)
        self.optimizer.zero_grad(set_to_none=True)
        loss = reward = 0.0
        for batch, size in zip(batches, sizes):
            result = adjoint_matching_loss(
                self.teacher, self.student, self.sampler, self.reward,
                to_device(batch, self.config.training.device),
                beta=self.config.training.beta, reward_input=self.config.reward.input,
                generator=self.generator,
            )
            if not torch.isfinite(result.loss):
                raise FloatingPointError("Nonfinite AM loss")
            weight = size / samples
            (result.loss * weight).backward()
            loss += result.loss.item() * weight
            reward += result.reward_values.mean().item() * weight
        self.optimizer.step()
        self.step += 1
        self.samples_seen += samples
        epoch, offset = self.data_position
        if offset + len(batches) == len(self.data.train_loader):
            self.data_position = (epoch + 1, 0)
        else:
            self.data_position = (epoch, offset + len(batches))
        return {"loss": loss, "reward": reward}

    def _log(self, values):
        record = json.dumps(safe({"step": self.step, "samples_seen": self.samples_seen,
                                  "data_position": list(self.data_position), **values}), allow_nan=False)
        with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as file:
            print(record, file=file)
        print(record, flush=True)

    @property
    def last_path(self):
        return self.output / "checkpoints" / "last.pt"

    @property
    def best_path(self):
        return self.output / "best" / "reward" / "checkpoint.pt"

    def _validation_path(self, step):
        return self.output / "validation" / f"step_{step:08d}"

    def _state(self):
        return {
            "format_version": 1, "kind": "bridge_soc", "soc_protocol_version": 1,
            **self.provenance, "config": asdict(self.config),
            "student": self.student.state_dict(), "optimizer": self.optimizer.state_dict(),
            "step": self.step, "samples_seen": self.samples_seen, "data_position": self.data_position,
            "rng": capture_rng_state(), "rollout_rng": self.generator.get_state(),
            "best": self.best, "last_validation_step": self.last_validation_step,
            "evaluation_samples": self.evaluation_samples,
        }

    def _publish_best(self, state):
        # last.pt commits the selection first. If publishing is interrupted, it
        # contains the winning weights and points to the complete val report.
        self.best_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self._validation_path(state["step"]), self.best_path.parent, dirs_exist_ok=True)
        save_checkpoint(self.best_path, state)

    def _report(self, evaluator, split, step):
        report = evaluator.evaluate(self.student, self.teacher)
        ids = self.evaluation_samples[split]
        if (report.protocol["sample_ids"] != ids["sample_ids"]
                or report.protocol["visual_sample_ids"] != ids["visual_ids"]):
            raise ValueError(f"SOC {split} evaluation samples changed")
        report.protocol.update(step=step, split_checksum=self.provenance["split_checksum"],
                               teacher=self.provenance["teacher"], criterion="reward")
        return report

    def validate(self):
        if self.evaluator is None:
            self.evaluator = SOCEvaluator(self.data, self.sampler, self.reward, self.config, "val")
        report = self._report(self.evaluator, "val", self.step)
        report.save(self._validation_path(self.step))
        self.last_validation_step = self.step
        value = report.summary["restoration"]["reward"]["mean"]
        improved = self.best is None or value > self.best["value"]
        if improved:
            self.best = {"step": self.step, "value": value}
        self._log({"validation": report.summary, "best": self.best})
        state = self._state()
        save_checkpoint(self.last_path, state)
        if improved:
            self._publish_best(state)

    def test_best(self):
        self.evaluator = None
        state = load_soc_checkpoint(self.best_path)
        self.student.load_state_dict(state["student"])
        evaluator = SOCEvaluator(self.data, self.sampler, self.reward, self.config, "test")
        report = self._report(evaluator, "test", state["step"])
        report.protocol["selection"] = self.best
        report.save(self.output / "test" / "reward")

    def resume(self, path):
        state = load_soc_checkpoint(path)
        for key, expected in self.provenance.items():
            if state[key] != expected:
                raise ValueError(f"SOC resume mismatch for {key}")
        original, current = state["config"], asdict(self.config)
        for key in ("seed", "reward"):
            if original[key] != current[key]:
                raise ValueError(f"SOC resume mismatch for {key}")
        for key in ("beta", "lr", "batch_size", "device", "grad_accum_steps"):
            if original["training"].get(key, 1) != current["training"][key]:
                raise ValueError(f"SOC resume mismatch for training.{key}")
        self.step = state["step"]
        self.data_position = tuple(state["data_position"])
        if self.step > self.config.training.max_steps:
            raise ValueError("max_steps precedes the checkpoint")
        loader = self.data.train_loader
        epoch, offset = self.data_position
        accumulation = self.config.training.grad_accum_steps
        steps_per_pass = (len(loader) + accumulation - 1) // accumulation
        if (epoch < 0 or not 0 <= offset < len(loader) or offset % accumulation
                or epoch * steps_per_pass + offset // accumulation != self.step):
            raise ValueError("SOC resume mismatch for data position")
        self.samples_seen = epoch * len(loader.dataset) + offset * loader.batch_size
        if state.get("soc_protocol_version"):
            if (original["training"]["validation_interval"] != current["training"]["validation_interval"]
                    or original["evaluation"] != current["evaluation"]
                    or state["evaluation_samples"] != self.evaluation_samples):
                raise ValueError("SOC resume mismatch for evaluation protocol")
            if state["samples_seen"] != self.samples_seen:
                raise ValueError("SOC resume mismatch for samples_seen")
            self.best = state["best"]
            self.last_validation_step = state["last_validation_step"]
            if self.best:
                best_step = self.best["step"]
                report_path = self._validation_path(best_step)
                if not all((report_path / name).is_file()
                           for name in ("protocol.json", "summary.json", "per_image.jsonl")):
                    raise ValueError(f"Missing historical SOC validation report: {report_path}")
                if best_step == self.step:
                    self._publish_best(state)
                elif not self.best_path.is_file() or load_soc_checkpoint(self.best_path)["best"] != self.best:
                    raise ValueError(f"SOC resume mismatch for historical best: {self.best_path}")
        else:
            self.migrate_validation = True
        self.student.load_state_dict(state["student"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["rollout_rng"])
        restore_rng_state(state["rng"])
        self.resumed = True

    def _recover_log(self):
        path = self.output / "metrics.jsonl"
        if not path.exists():
            return
        with TemporaryDirectory(dir=self.output) as directory:
            temporary = Path(directory) / path.name
            with path.open() as source, temporary.open("w") as destination:
                for line in source:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    if (record["step"] <= self.step and
                            ("validation" not in record or
                             (self.last_validation_step is not None
                              and record["step"] <= self.last_validation_step))):
                        destination.write(line)
            temporary.replace(path)

    def fit(self):
        loader = self.data.train_loader
        cfg = self.config.training
        if self.resumed:
            self._recover_log()
        if self.migrate_validation:
            self.validate()
        while self.step < cfg.max_steps:
            epoch, offset = self.data_position
            loader.sampler.set_epoch(epoch)
            loader.generator.manual_seed(self.config.seed + epoch)
            batches = islice(loader, offset, None)
            while self.step < cfg.max_steps:
                group = list(islice(batches, cfg.grad_accum_steps))
                if not group:
                    break
                metrics = self.train_step(group)
                if self.step == 1 or self.step % cfg.log_interval == 0:
                    self._log(metrics)
                if self.step % cfg.validation_interval == 0 or self.step == cfg.max_steps:
                    self.validate()
                elif self.step % cfg.checkpoint_interval == 0:
                    save_checkpoint(self.last_path, self._state())
        if self.last_validation_step != self.step:
            self.validate()
        self.test_best()
