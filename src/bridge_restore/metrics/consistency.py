import torch
from .pixel import PixelMetric
from .base import MetricBatch

class ObservationErrorMetric(PixelMetric):
    def __init__(self, corruption, kind): super().__init__(kind); self.corruption = corruption
    def values_for(self, batch):
        if batch.observation is None: raise ValueError("Observation required")
        degraded = torch.stack([self.corruption(((x.detach().cpu()+1)/2).clamp(0,1))*2-1 for x in batch.prediction]).to(batch.prediction.device)
        return super().values_for(MetricBatch(degraded, batch.observation, batch.sample_ids))
