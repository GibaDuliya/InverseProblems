from .base import MetricResult

class DistributionMetric:
    """TorchMetrics Inception backend; explicitly distinct from Clean-FID."""
    def __init__(self, kind, device, **kwargs):
        if kind == "fid":
            from torchmetrics.image.fid import FrechetInceptionDistance
            cls = FrechetInceptionDistance
        else:
            from torchmetrics.image.kid import KernelInceptionDistance
            cls = KernelInceptionDistance
        self.kind, self.kwargs = kind, kwargs
        self.metric = cls(normalize=True, **kwargs).to(device)
    def reset(self): self.metric.reset()
    def update(self, batch):
        self.metric.update(((batch.target.float()+1)/2).clamp(0,1), real=True)
        self.metric.update(((batch.prediction.float()+1)/2).clamp(0,1), real=False)
    def compute(self):
        value = self.metric.compute()
        return MetricResult({"mean": float(value[0]), "subset_std": float(value[1])} if self.kind == "kid" else {"mean": float(value)})
    def description(self): return {"implementation": "torchmetrics/torch-fidelity", "kind": self.kind, "parameters": self.kwargs}
