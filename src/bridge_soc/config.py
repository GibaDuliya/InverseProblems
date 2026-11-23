"""Configuration for initializing a SOC control denoiser."""
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Optional, Union

import yaml


@dataclass(frozen=True)
class ControlConfig:
    teacher_checkpoint: str
    teacher_weights: str = "ema"
    mode: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.teacher_checkpoint, str) or not self.teacher_checkpoint.strip():
            raise ValueError("teacher_checkpoint must be a nonempty path")
        if self.teacher_weights not in ("ema", "raw"):
            raise ValueError("teacher_weights must be ema or raw")
        if self.mode not in (None, "bm", "abm"):
            raise ValueError("mode must be bm, abm, or null")


def load_control_config(path: Union[str, Path]) -> ControlConfig:
    with Path(path).open(encoding="utf-8") as file:
        values = yaml.safe_load(file)

    if not isinstance(values, dict):
        raise ValueError("Control configuration must be a YAML mapping")

    return ControlConfig(**values)


@dataclass(frozen=True)
class SOCSamplingConfig:
    num_steps: Optional[int] = None
    start_offset: Optional[float] = None
    clip_output: Optional[bool] = None

    def __post_init__(self):
        if self.num_steps is not None:
            _positive_int(self.num_steps, "sampling.num_steps")
        if self.start_offset is not None:
            _positive_real(self.start_offset, "sampling.start_offset")
        if self.clip_output is not None and type(self.clip_output) is not bool:
            raise ValueError("sampling.clip_output must be a boolean")


@dataclass(frozen=True)
class RewardConfig:
    name: str
    input: str

    def __post_init__(self):
        if self.name != "jpeg_consistency":
            raise ValueError("Unknown reward name")
        if self.input not in ("raw", "clipped"):
            raise ValueError("reward.input must be raw or clipped")


@dataclass(frozen=True)
class SOCTrainingConfig:
    beta: float
    lr: float = 1e-5
    device: str = "cuda"
    batch_size: int = 1
    grad_accum_steps: int = 1
    max_steps: Optional[int] = None
    log_interval: int = 1
    validation_interval: int = 100
    checkpoint_interval: int = 100

    def __post_init__(self):
        _positive_real(self.beta, "training.beta")
        _positive_real(self.lr, "training.lr")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("training.device must be cpu or cuda")
        if self.max_steps is not None:
            _positive_int(self.max_steps, "training.max_steps")
        for name in ("batch_size", "grad_accum_steps", "log_interval",
                     "validation_interval", "checkpoint_interval"):
            _positive_int(getattr(self, name), f"training.{name}")


EVALUATION_FIELDS = {
    "subset_size", "subset_seed", "batch_size", "metrics", "input_baseline",
    "test_subset_size", "test_metrics", "visual_samples_path", "test_visual_samples_path",
}


@dataclass(frozen=True)
class SOCConfig:
    control: ControlConfig
    reward: RewardConfig
    training: SOCTrainingConfig
    sampling: SOCSamplingConfig = field(default_factory=SOCSamplingConfig)
    evaluation: dict = field(default_factory=dict)
    seed: int = 42
    output_dir: str = "runs/soc/jpeg"

    def __post_init__(self):
        if not isinstance(self.evaluation, dict) or self.evaluation.keys() - EVALUATION_FIELDS:
            raise ValueError("Unknown or invalid SOC evaluation override")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32 - 2:
            raise ValueError("seed must be a nonnegative 32-bit seed")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a nonempty path")


def _positive_real(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def load_soc_config(path):
    with Path(path).open(encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict):
        raise ValueError("SOC configuration must be a YAML mapping")
    if isinstance(values.get("training"), dict) and "validation_batches" in values["training"]:
        raise ValueError("Replace training.validation_batches with evaluation.subset_size (images)")
    sections = {"control": ControlConfig, "reward": RewardConfig,
                "training": SOCTrainingConfig, "sampling": SOCSamplingConfig}
    for name, schema in sections.items():
        if name in values:
            if not isinstance(values[name], dict):
                raise ValueError(f"{name} must be a YAML mapping")
            values[name] = schema(**values[name])
    return SOCConfig(**values)
