"""Deterministic, shape-preserving degradations on CPU RGB tensors in [0,1]."""
from io import BytesIO
import importlib
from typing import Protocol
import numpy as np
from PIL import Image
import torch
from .datasets import validate_image


class Corruption(Protocol):
    def __call__(self, image: torch.Tensor) -> torch.Tensor: ...
    def to_config(self) -> dict: ...


def resolve_factory(name):
    module, attribute = name.split(":", 1)
    return getattr(importlib.import_module(module), attribute)


class JPEGCorruption:
    """Real Pillow JPEG round trip. Quantization: round(255*x) to uint8.

    quality and subsampling are fixed attributes. Decoding returns RGB float32.
    Bit-identical results across different codec/Pillow versions are not promised.
    """
    def __init__(self, quality=10, subsampling=2, optimize=False, progressive=False):
        if not isinstance(quality, int) or not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be an integer in [1,100]")
        if subsampling not in (0, 1, 2):
            raise ValueError("JPEG subsampling must be 0, 1 or 2")
        self.quality = quality
        self.subsampling = subsampling
        self.optimize = optimize
        self.progressive = progressive

    def to_config(self):
        return {"quality": self.quality, "subsampling": self.subsampling,
                "optimize": self.optimize, "progressive": self.progressive}

    def encode(self, image):
        validate_image(image)
        array = image.detach().mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        buffer = BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", **self.to_config())
        return buffer.getvalue()

    @staticmethod
    def decode(encoded):
        with Image.open(BytesIO(encoded)) as decoded:
            result = np.array(decoded.convert("RGB"), dtype=np.float32, copy=True)
        return torch.from_numpy(result).permute(2, 0, 1).contiguous() / 255


    def __call__(self, image):
        return self.decode(self.encode(image))


def build_corruption(config):
    corruption = resolve_factory(config.factory)(**config.kwargs)
    if not callable(corruption) or not callable(getattr(corruption, "to_config", None)):
        raise TypeError("Corruption must implement __call__ and to_config")
    return corruption
