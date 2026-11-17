from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EvaluationConfig:
    enabled: bool = False
    weights: str = "ema"
    unit: str = "step"
    interval: int = 500
    subset_size: Optional[int] = 512
    subset_seed: int = 42
    batch_size: int = 8
    metrics: dict = field(default_factory=lambda: {"psnr": {}})
    criteria: dict = field(default_factory=dict)
    selection_metric: str = "psnr"
    selection_mode: str = "max"
    input_baseline: bool = True
    final_test: bool = False
    test_subset_size: Optional[int] = None
    test_metrics: Optional[dict] = None
    visual_samples_path: Optional[str] = None
    test_visual_samples_path: Optional[str] = None

    def __post_init__(self):
        if self.weights not in ("raw", "ema") or self.unit not in ("step", "epoch"): raise ValueError("Invalid evaluation weights/unit")
        if self.interval < 1 or self.batch_size < 1: raise ValueError("Invalid evaluation interval/batch size")
        if any(x is not None and x < 1 for x in (self.subset_size, self.test_subset_size)): raise ValueError("Invalid subset size")
        if not self.criteria and (self.selection_metric not in self.metrics or self.selection_mode not in ("min", "max")):
            raise ValueError("Invalid selection criterion")
        from .selection import validate_criteria
        validate_criteria(self.resolved_criteria(), self.metrics)

    def resolved_criteria(self):
        # Old configs/checkpoints remain readable; new configs use criteria only.
        return self.criteria or {self.selection_metric: {
            "metric": self.selection_metric, "mode": self.selection_mode, "test": self.final_test}}
