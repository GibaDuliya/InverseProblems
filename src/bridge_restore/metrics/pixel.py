from .base import Metric

class PixelMetric(Metric):
    def __init__(self, kind): self.kind = kind; self.reset()
    def values_for(self, batch):
        delta = (batch.prediction.float() - batch.target.float()) / 2
        error = delta.abs() if self.kind == "mae" else delta.square()
        value = error.flatten(1).mean(1)
        return -10 * value.log10() if self.kind == "psnr" else value
    def description(self): return {"implementation": "torch", "kind": self.kind, "data_range": 1}
