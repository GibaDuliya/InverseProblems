"""Measure SOC reward and restoration metrics on the same reconstructions."""
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from bridge_restore.checkpoint import capture_rng_state, restore_rng_state
from bridge_restore.evaluation.config import EvaluationConfig
from bridge_restore.evaluation.evaluator import RestorationEvaluator, evaluation_subset
from bridge_restore.sampling import SamplingResult

from .adjoint_matching import reward_image
from .config import EVALUATION_FIELDS


def resolve_evaluation(base, config):
    profile = asdict(base.evaluation)
    if not base.evaluation.enabled and base.data.factory.endswith(":CelebADataset"):
        path = Path("configs/celeba_medium_abm_T1.yaml")
        profile = yaml.safe_load(path.read_text())["evaluation"]
    options = {key: profile[key] for key in EVALUATION_FIELDS & profile.keys()}
    options.update(config.evaluation)
    for key in ("subset_size", "test_subset_size", "batch_size"):
        value = options.get(key)
        if (value is not None or key == "batch_size") and (type(value) is not int or value < 1):
            raise ValueError(f"evaluation.{key} must be a positive integer or a supported null")
    seed = options["subset_seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("evaluation.subset_seed must be a nonnegative 32-bit seed")
    for key in ("metrics", "test_metrics"):
        metrics = options.get(key)
        if key == "test_metrics" and metrics is None:
            continue
        if not isinstance(metrics, dict) or not metrics or "reward" in metrics:
            raise ValueError(f"evaluation.{key} must specify restoration metrics")
    return options


def split_config(config, split):
    # EvaluationConfig validates a baseline criterion. SOC selection is handled by the trainer.
    profile = EvaluationConfig(**config.evaluation, enabled=True, weights="raw",
                               interval=config.training.validation_interval,
                               selection_metric=next(iter(config.evaluation["metrics"])))
    if split == "val":
        return profile
    metrics = profile.test_metrics or profile.metrics
    return replace(profile, subset_size=profile.test_subset_size, metrics=metrics,
                   visual_samples_path=profile.test_visual_samples_path,
                   selection_metric=next(iter(metrics)))


def evaluation_ids(dataset, config, split):
    profile = split_config(config, split)
    indices, visual_ids = evaluation_subset(dataset, profile, split)
    if "kid" in profile.metrics and len(indices) < (profile.metrics["kid"] or {}).get("subset_size", 1000):
        raise ValueError(f"{split} subset is too small for KID")
    return {"sample_ids": [dataset.sample_ids[i] for i in indices], "visual_ids": visual_ids}


@contextmanager
def isolated_rng(seed):
    state = capture_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        restore_rng_state(state)


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item, device) for item in value)
    return value


class SOCEvaluator:
    def __init__(self, data, sampler, reward, config, split):
        self.device = config.training.device
        self.sampler, self.reward, self.config = sampler, reward, config
        self.seed = (config.seed + (1 if split == "val" else 2)) % 2**32
        self.split = split
        self.teacher_rewards = {}
        dataset = (data.val_loader if split == "val" else data.test_loader).dataset
        with isolated_rng(self.seed):
            self.evaluator = RestorationEvaluator(dataset, data.corruption, sampler,
                                                  split_config(config, split), self.device, split)

    def _rewards(self, terminal, batch):
        values = self.reward(reward_image(terminal, self.config.reward.input), batch)
        if values.shape != (len(terminal),) or not torch.isfinite(values).all():
            raise ValueError("Evaluation reward must return one finite scalar per image")
        return values.detach().cpu().tolist()

    def evaluate(self, student, teacher):
        def generator():
            return torch.Generator(device=self.device).manual_seed(self.seed)

        student_rng, teacher_rng = generator(), generator()
        student_rewards = {}
        teacher_rewards = {}

        def sample_batch(model, batch, observation):
            # x0 and xT are already handled by RestorationEvaluator.
            metadata = {key: to_device(value, self.device) for key, value in batch.items()
                        if key not in ("x0", "xT")}
            metadata.update(xT=observation, x0=batch["x0"].to(self.device))
            ids = batch["sample_id"]
            result = self.sampler.sample(model, observation, student_rng, clip_output=False)
            terminal = result.reconstruction
            student_rewards.update(zip(ids, self._rewards(terminal, metadata)))
            if not self.teacher_rewards:
                reference = self.sampler.sample(teacher, observation, teacher_rng, clip_output=False)
                teacher_rewards.update(zip(ids, self._rewards(reference.reconstruction, metadata)))
            reconstruction = terminal.clamp(-1, 1) if self.sampler.clip_output else terminal
            return SamplingResult(reconstruction, result.nfe)

        with isolated_rng(self.seed):
            report = self.evaluator.evaluate(student, sample_batch=sample_batch)
        if not self.teacher_rewards:
            self.teacher_rewards = teacher_rewards
        count = report.summary["samples"]
        mean = sum(student_rewards.values()) / count
        teacher_mean = sum(self.teacher_rewards.values()) / count
        report.summary["restoration"]["reward"] = {"mean": mean}
        report.summary.update(teacher_reward_mean=teacher_mean, reward_gain=mean - teacher_mean)
        for row in report.per_image:
            key = row["sample_id"]
            row.update({"restoration/reward": student_rewards[key],
                        "teacher/reward": self.teacher_rewards[key]})
        report.protocol.update(split=self.split, weights="raw", nfe=report.summary["nfe"],
                               reward=asdict(self.config.reward), evaluation_seed=self.seed,
                               evaluation_noise="fixed per split and batch order, shared with teacher")
        return report
