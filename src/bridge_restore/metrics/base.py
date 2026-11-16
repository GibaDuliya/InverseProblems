"""Metrics consume RGB BCHW floats in [-1, 1]; observation is optional."""
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class MetricBatch:
    prediction: torch.Tensor
    target: torch.Tensor
    sample_ids: list
    observation: Optional[torch.Tensor] = None

@dataclass
class MetricResult:
    summary: dict
    per_image: Optional[dict] = None

class Metric:
    def reset(self): self.values = {}
    def update(self, batch):
        values = self.values_for(batch).detach().cpu().reshape(-1)
        if len(values) != len(batch.sample_ids): raise ValueError("Metric must return one value per image")
        for key, value in zip(batch.sample_ids, values):
            if key in self.values: raise ValueError("Duplicate sample ID")
            self.values[key] = float(value)
    def compute(self):
        if not self.values: raise ValueError("Empty evaluation")
        return MetricResult({"mean": sum(self.values.values()) / len(self.values)}, dict(self.values))
    def description(self): return {"implementation": type(self).__name__}
