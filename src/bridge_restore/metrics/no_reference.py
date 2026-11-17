"""Native-resolution RGB MUSIQ KonIQ-10k; no external resize or crop."""
import importlib.metadata
import torch
from .base import Metric


class MUSIQMetric(Metric):
    def __init__(self, device, batch_size=1):
        import pyiqa
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("MUSIQ batch_size must be positive")
        self.network = pyiqa.create_metric("musiq", device=device).eval().requires_grad_(False)
        self.batch_size = batch_size
        self.reset()

    @torch.no_grad()
    def values_for(self, batch):
        images = (batch.prediction.float() + 1).div(2).clamp(0, 1)
        values = torch.cat([self.network(part).reshape(-1) for part in images.split(self.batch_size)])
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite MUSIQ scores")
        return values

    def description(self):
        return {"implementation": "pyiqa", "version": importlib.metadata.version("pyiqa"),
                "model": "musiq", "weights": "KonIQ-10k", "input": "RGB [0,1], native resolution"}
