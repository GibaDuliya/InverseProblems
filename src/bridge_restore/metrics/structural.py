from .base import Metric

class StructuralMetric(Metric):
    def __init__(self, kind, **kwargs):
        from torchmetrics.functional.image import structural_similarity_index_measure, multiscale_structural_similarity_index_measure
        self.fn = structural_similarity_index_measure if kind == "ssim" else multiscale_structural_similarity_index_measure
        self.kwargs = kwargs
        self.reset()
    def values_for(self, batch):
        return self.fn((batch.prediction.float()+1)/2, (batch.target.float()+1)/2,
                       data_range=1.0, reduction="none", **self.kwargs)
    def description(self): return {"implementation": self.fn.__name__, "parameters": self.kwargs}
