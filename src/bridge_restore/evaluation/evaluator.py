from dataclasses import asdict
from copy import copy
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from ..metrics import Metric, MetricBatch, build_metrics
from .report import EvaluationReport
from .visualization import load_visual_samples

class RestorationEvaluator:
    """Persistent metrics and fixed subset; generation consumes global RNG."""
    def __init__(self, dataset, corruption, sampler, config, device, split="val"):
        self.config, self.sampler, self.device = config, sampler, device
        self.split = split
        n = len(dataset)
        if not n: raise ValueError("Empty evaluation dataset")
        if config.subset_size is not None and config.subset_size > n: raise ValueError("Subset exceeds dataset size")
        self.visual_ids = load_visual_samples(config.visual_samples_path, split)
        visual_indices = []
        if self.visual_ids:
            if not hasattr(dataset, "sample_ids"): raise ValueError("Visual samples require dataset.sample_ids metadata")
            lookup = {key: i for i, key in enumerate(dataset.sample_ids)}
            missing = set(self.visual_ids) - set(lookup)
            if missing: raise ValueError(f"Visual IDs absent from {split}: {sorted(missing)}")
            visual_indices = [lookup[key] for key in self.visual_ids]
        if config.subset_size is not None and config.subset_size < len(visual_indices):
            raise ValueError("Evaluation subset smaller than visual panel")
        if config.subset_size is None:
            self.indices = list(range(n))
        else:
            excluded = set(visual_indices)
            remaining = [i for i in range(n) if i not in excluded]
            permutation = torch.randperm(len(remaining), generator=torch.Generator().manual_seed(config.subset_seed)).tolist()
            self.indices = visual_indices + [remaining[i] for i in permutation[:config.subset_size-len(visual_indices)]]
        self.loader = DataLoader(Subset(dataset, self.indices), batch_size=config.batch_size, shuffle=False)
        self.metrics = build_metrics(config.metrics, device, corruption)
        # Independent accumulators, shared frozen networks and reference caches.
        self.baseline = {}
        if config.input_baseline:
            for name, metric in self.metrics.items():
                if name.startswith("observation_"):
                    continue
                # Scalar metrics accumulate only in values; distribution metrics
                # also own mutable TorchMetrics state and need their own backend.
                self.baseline[name] = (copy(metric) if isinstance(metric, Metric) else
                    build_metrics({name: config.metrics[name]}, device, corruption)[name])

    @torch.no_grad()
    def evaluate(self, model):
        was_training = model.training
        model.eval()
        groups = {"restoration": self.metrics, "input": self.baseline}
        for metrics in groups.values():
            for metric in metrics.values(): metric.reset()
        ids, nfe = [], None
        examples = {}
        visual_set = set(self.visual_ids)
        try:
            for batch in tqdm(self.loader, desc=self.split.capitalize(), unit="batch", leave=False):
                x0, xT = batch["x0"].to(self.device), batch["xT"].to(self.device)
                result = self.sampler.sample(model, xT)
                nfe = result.nfe
                for i, key in enumerate(batch["sample_id"]):
                    if key in visual_set:
                        examples[key] = tuple(x[i].detach().cpu().clone() for x in (x0, xT, result.reconstruction))
                ids.extend(batch["sample_id"])
                for group, metrics in groups.items():
                    mb = MetricBatch(result.reconstruction if group == "restoration" else xT, x0, batch["sample_id"], xT)
                    if not torch.isfinite(mb.prediction).all(): raise FloatingPointError("Nonfinite predictions")
                    for metric in metrics.values(): metric.update(mb)
            rows = {key: {"sample_id": key} for key in ids}
            summary = {"samples": len(ids), "nfe": nfe}
            for group, metrics in groups.items():
                summary[group] = {}
                for name, metric in metrics.items():
                    value = metric.compute(); summary[group][name] = value.summary
                    for key, number in (value.per_image or {}).items(): rows[key][group+"/"+name] = number
            return EvaluationReport(summary, list(rows.values()), {"config": asdict(self.config), "sample_ids": ids,
                "visual_samples_path": self.config.visual_samples_path, "visual_sample_ids": self.visual_ids,
                "metrics": {k:v.description() for k,v in self.metrics.items()}, "torch": torch.__version__,
                "sampler": {"num_steps": self.sampler.num_steps, "start_offset": self.sampler.start_offset, "clip_output": self.sampler.clip_output}},
                {key: examples[key] for key in self.visual_ids})
        finally: model.train(was_training)
