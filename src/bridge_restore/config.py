"""Strict YAML configuration. Paths are relative to the working directory."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional
import math
import yaml
from .evaluation.config import EvaluationConfig


@dataclass
class DataConfig:
    factory: str = "bridge_restore.data.datasets:CelebADataset"
    kwargs: dict = field(default_factory=dict)
    image_size: int = 128
    crop_mode: str = "center_square"
    interpolation: str = "bicubic"
    antialias: bool = True
    num_workers: int = 0


@dataclass
class SplitConfig:
    mode: str = "official"
    manifest_path: Optional[str] = None
    group_by: str = "identity"
    fractions: list = field(default_factory=lambda: [0.8, 0.1, 0.1])
    seed: int = 42


@dataclass
class CorruptionConfig:
    factory: str = "bridge_restore.data.corruptions:JPEGCorruption"
    kwargs: dict = field(default_factory=lambda: {"quality": 10, "subsampling": 2})


@dataclass
class ModelConfig:
    mode: str = "abm"
    channels: int = 128
    num_res_blocks: int = 2
    channel_mult: list = field(default_factory=lambda: [1, 1, 2, 3, 4])
    attention_resolutions: list = field(default_factory=lambda: [16, 8])
    num_heads: int = 4
    dropout: float = 0.1
    use_checkpoint: bool = False


@dataclass
class BridgeConfig:
    sigma_max: float = 1.0
    sigma_min: float = 0.001
    sigma_data: float = 0.5
    cov_xy: float = 0.0


@dataclass
class TrainingConfig:
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: Optional[int] = 10000
    max_epochs: Optional[int] = None
    lr: float = 1e-4
    weight_decay: float = 0.0
    precision: str = "float32"
    device: str = "cuda"
    ema_decay: float = 0.999
    validation_interval: int = 500
    validation_batches: int = 10
    checkpoint_interval: int = 1000
    log_interval: int = 50


@dataclass
class SamplingConfig:
    num_steps: int = 100
    start_offset: float = 0.0001
    clip_output: bool = True


@dataclass
class ExperimentConfig:
    seed: int = 42
    output_dir: str = "runs/experiment"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


SECTIONS = {"data": DataConfig, "split": SplitConfig,
            "corruption": CorruptionConfig, "model": ModelConfig,
            "bridge": BridgeConfig, "training": TrainingConfig,
            "sampling": SamplingConfig, "evaluation": EvaluationConfig}


def _construct(cls, values):
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def config_from_dict(values) -> ExperimentConfig:
    values = dict(values)
    for key, cls in SECTIONS.items():
        if key in values:
            values[key] = _construct(cls, values[key])
    config = _construct(ExperimentConfig, values)
    validate_config(config)
    return config


def load_config(path) -> ExperimentConfig:
    with open(path) as f:
        return config_from_dict(yaml.safe_load(f) or {})


def save_config(config, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))


def validate_config(c):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    def positive_int(v):
        return isinstance(v, int) and not isinstance(v, bool) and v > 0
    require(c.model.mode in ("bm", "abm"), "model.mode must be bm or abm")
    require(c.split.mode in ("load", "official", "create_if_missing"), "Invalid split.mode")
    if c.split.mode != "official":
        require(bool(c.split.manifest_path), "Custom splits require manifest_path")
        require(c.split.group_by in ("identity", "sample"), "Invalid split.group_by")
    if c.split.mode == "create_if_missing":
        require(len(c.split.fractions) == 3 and all(0 < x < 1 for x in c.split.fractions)
                and math.isclose(sum(c.split.fractions), 1), "Split fractions must be positive and sum to 1")
    require(c.data.crop_mode in ("center_square", "none"), "Invalid crop_mode")
    require(c.data.interpolation in ("bilinear", "bicubic"), "Invalid interpolation")
    require(positive_int(c.data.image_size), "image_size must be positive")
    require(isinstance(c.data.num_workers, int) and c.data.num_workers >= 0, "Invalid num_workers")
    require(positive_int(c.model.channels) and c.model.channels % 32 == 0, "channels must be a positive multiple of 32")
    require(bool(c.model.channel_mult) and all(positive_int(x) for x in c.model.channel_mult), "Invalid channel_mult")
    require(positive_int(c.model.num_res_blocks) and positive_int(c.model.num_heads), "Invalid blocks/heads")
    require(all(c.model.channels * m % c.model.num_heads == 0 for m in c.model.channel_mult), "Channels must be divisible by heads")
    require(c.data.image_size % 2 ** (len(c.model.channel_mult) - 1) == 0, "image_size incompatible with UNet depth")
    valid_res = {c.data.image_size // 2 ** i for i in range(len(c.model.channel_mult))}
    require(set(c.model.attention_resolutions) <= valid_res, "Attention resolutions must be UNet resolutions")
    require(0 <= c.model.dropout < 1, "Invalid dropout")
    b = c.bridge
    require(all(math.isfinite(x) for x in (b.sigma_min, b.sigma_max, b.sigma_data, b.cov_xy)), "Nonfinite bridge parameters")
    require(0 < b.sigma_min < b.sigma_max and b.sigma_data > 0, "Invalid bridge scales")
    require(abs(b.cov_xy) < b.sigma_data ** 2, "Require abs(cov_xy) < sigma_data²")
    require(0 < c.sampling.start_offset < b.sigma_max - b.sigma_min, "Invalid absolute start_offset")
    require(positive_int(c.sampling.num_steps), "num_steps must be positive")
    for key in ("batch_size", "grad_accum_steps", "validation_interval",
                "validation_batches", "checkpoint_interval", "log_interval"):
        require(positive_int(getattr(c.training, key)), f"training.{key} must be positive")
    require(c.training.max_epochs is not None or c.training.max_steps is not None, "Set max_epochs or max_steps")
    for key in ("max_epochs", "max_steps"):
        value = getattr(c.training, key)
        require(value is None or positive_int(value), f"training.{key} must be positive or null")
    require(math.isfinite(c.training.lr) and c.training.lr > 0, "Invalid lr")
    require(math.isfinite(c.training.weight_decay) and c.training.weight_decay >= 0, "Invalid weight_decay")
    require(0 <= c.training.ema_decay < 1, "Invalid ema_decay")
    require(c.training.precision in ("float32", "float16", "bfloat16"), "Invalid precision")
    require(c.training.device in ("cpu", "cuda"), "device must be cpu or cuda")
    require(c.training.device == "cuda" or c.training.precision == "float32", "CPU training uses float32")
