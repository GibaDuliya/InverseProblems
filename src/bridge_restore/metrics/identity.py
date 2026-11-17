"""ArcFace identity cosine using target landmarks for both aligned crops.

Reference caches are scoped to one evaluator/dataset and survive reset(). Missing
reference detections are reported, not assigned a made-up identity score.
InsightFace/ONNX runs on CPU by default, independently of the training device.
"""
import hashlib
import importlib.metadata
from pathlib import Path
import numpy as np
import torch
from .base import Metric, MetricResult


class ArcFaceMetric(Metric):
    def __init__(self, device, root="~/.insightface", det_size=320,
                 det_thresh=0.5, min_coverage=0.9):
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align
        if not 0 <= min_coverage <= 1 or not 0 < det_thresh < 1 or det_size < 1:
            raise ValueError("Invalid ArcFace detection/coverage configuration")
        self.app = FaceAnalysis(name="buffalo_l", root=str(Path(root).expanduser()),
            allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size), det_thresh=det_thresh)
        self.recognition = self.app.models["recognition"]
        if tuple(self.recognition.input_size) != (112, 112):
            raise ValueError("Expected ArcFace input size 112x112")
        self.align = face_align
        self.references = {}
        self.min_coverage, self.det_size, self.det_thresh = min_coverage, det_size, det_thresh
        self.model_hash = hashlib.sha256(Path(self.recognition.model_file).read_bytes()).hexdigest()
        self.reset()

    @staticmethod
    def _bgr(images):
        return images.detach().float().cpu().add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()[..., ::-1].copy()

    def _reference(self, key, image):
        if key not in self.references:
            boxes, landmarks = self.app.det_model.detect(image, max_num=0)
            if len(boxes) == 0:
                self.references[key] = None
            else:
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                kps = landmarks[area.argmax()]
                crop = self.align.norm_crop(image, landmark=kps, image_size=112)
                embedding = self.recognition.get_feat(crop).reshape(-1)
                embedding = embedding / max(np.linalg.norm(embedding), 1e-12)
                self.references[key] = (kps, embedding)
        return self.references[key]

    @torch.no_grad()
    def values_for(self, batch):
        predictions, targets = self._bgr(batch.prediction), self._bgr(batch.target)
        values = np.full(len(predictions), np.nan, dtype=np.float32)
        crops, references, indices = [], [], []
        for i, key in enumerate(batch.sample_ids):
            reference = self._reference(key, targets[i])
            if reference is not None:
                kps, embedding = reference
                crops.append(self.align.norm_crop(predictions[i], landmark=kps, image_size=112))
                references.append(embedding)
                indices.append(i)
        if crops:
            embeddings = self.recognition.get_feat(crops)
            embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
            scores = (embeddings * np.asarray(references)).sum(1)
            if not np.isfinite(scores).all():
                raise FloatingPointError("Nonfinite ArcFace embeddings")
            values[indices] = np.clip(scores, -1, 1)
        return torch.from_numpy(values)

    def compute(self):
        if not self.values:
            raise ValueError("Empty evaluation")
        valid = {key: value for key, value in self.values.items() if np.isfinite(value)}
        missing = [key for key in self.values if key not in valid]
        coverage = len(valid) / len(self.values)
        if not valid or coverage < self.min_coverage:
            raise ValueError(f"ArcFace reference coverage {coverage:.1%} below required {self.min_coverage:.1%}; missing IDs: {missing}")
        return MetricResult({"mean": sum(valid.values()) / len(valid), "coverage": coverage,
            "valid_samples": len(valid), "missing_sample_ids": missing},
            {key: value if key in valid else None for key, value in self.values.items()})

    def description(self):
        return {"implementation": "insightface", "version": importlib.metadata.version("insightface"),
            "model": "buffalo_l/w600k_r50.onnx", "sha256": self.model_hash,
            "alignment": "target landmarks, largest detected face; shared warp", "input_size": 112,
            "providers": ["CPUExecutionProvider"], "det_size": self.det_size,
            "det_thresh": self.det_thresh, "min_coverage": self.min_coverage}
