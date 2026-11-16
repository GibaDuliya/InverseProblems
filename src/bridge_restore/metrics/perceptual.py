from .base import Metric

class LPIPSMetric(Metric):
    def __init__(self, device, backbone="alex", version="0.1"):
        import lpips
        self.network = lpips.LPIPS(net=backbone, version=version).to(device).eval().requires_grad_(False)
        self.backbone, self.version = backbone, version
        self.reset()
    def values_for(self, batch): return self.network(batch.prediction.float(), batch.target.float()).flatten()
    def description(self): return {"implementation": "lpips", "backbone": self.backbone, "version": self.version}
