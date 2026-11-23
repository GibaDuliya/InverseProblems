"""Clean images and the sole conversion from [0,1] pairs to [-1,1]."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class CleanSample:
    """CPU float32 RGB in [0,1]; IDs must agree with dataset metadata."""
    image: torch.Tensor
    sample_id: str
    group_id: Optional[str] = None


class CleanImageDataset(Protocol):
    """Metadata are available without decoding images; order matches indexing."""
    sample_ids: Sequence[str]
    group_ids: Sequence[Optional[str]]

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> CleanSample: ...


def read_rgb(path):
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous() / 255


def validate_image(image):
    if (not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3
            or image.dtype != torch.float32 or image.device.type != "cpu"
            or min(image.shape[1:]) < 1):
        raise ValueError("Expected nonempty CPU float32 RGB tensor [3,H,W]")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Image must be finite and in [0,1]")


class CelebADataset(Dataset):
    """Aligned CelebA with its official, identity-disjoint partition.

    The partition file defines the sample IDs. Identity annotations are optional
    metadata for analysis/custom repartitioning, not needed for normal training.
    No download, crop or degradation happens here.
    """
    def __init__(self, root, images_dir="img_align_celeba",
                 identity_file=None, partition_file="list_eval_partition.txt"):
        self.root = Path(root)
        def rows(filename):
            result = {}
            for line in (self.root / filename).read_text().splitlines():
                if not line.strip():
                    continue
                name, value = line.split()
                if name in result:
                    raise ValueError(f"Duplicate annotation: {name}")
                result[name] = value
            return result
        partitions = rows(partition_file)
        names = {"0": "train", "1": "val", "2": "test"}
        self.sample_ids = sorted(partitions)
        self.official_split_assignments = {key: names[value] for key, value in partitions.items()}
        self.group_ids = [None] * len(self.sample_ids)
        if identity_file is not None:
            identities = rows(identity_file)
            if set(identities) != set(partitions):
                raise ValueError("CelebA partition and identity ID sets differ")
            self.group_ids = [identities[key] for key in self.sample_ids]
        self.image_paths = [self.root / images_dir / key for key in self.sample_ids]
        missing = next((p for p in self.image_paths if not p.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(missing)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        return CleanSample(read_rgb(self.image_paths[index]), self.sample_ids[index], self.group_ids[index])


class ImageFolderDataset(Dataset):
    """Simple custom-data example. Optional identity mapping is {relative_path: group_id}."""
    def __init__(self, root, groups_file=None):
        import json
        self.root = Path(root)
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        self.sample_ids = sorted(p.relative_to(self.root).as_posix()
                                 for p in self.root.rglob("*") if p.suffix.lower() in extensions)
        groups = json.loads(Path(groups_file).read_text()) if groups_file else {}
        self.group_ids = [groups.get(key) for key in self.sample_ids]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        return CleanSample(read_rgb(self.root / self.sample_ids[index]), self.sample_ids[index], self.group_ids[index])


class ImagePreprocessor:
    """Center-square crop (or no crop), then resize, before constructing observation."""
    def __init__(self, image_size, crop_mode="center_square", interpolation="bicubic", antialias=True):
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.interpolation = interpolation
        self.antialias = antialias

    def __call__(self, image):
        validate_image(image)
        if self.crop_mode == "center_square":
            h, w = image.shape[-2:]
            side = min(h, w)
            image = image[:, (h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side]
        elif self.crop_mode != "none":
            raise ValueError(f"Unknown crop mode: {self.crop_mode}")
        result = F.interpolate(image[None], size=(self.image_size, self.image_size),
                               mode=self.interpolation, align_corners=False, antialias=self.antialias)[0]
        return result.clamp(0, 1).contiguous()


class RestorationDataset(Dataset):
    """Selected clean IDs -> preprocess -> corruption -> paired normalization.

    Corruption must be deterministic, shape preserving and must not mutate its input.
    Factory datasets must return metadata consistent with sample_ids/group_ids.
    """
    def __init__(self, clean_dataset, indices, preprocessor, corruption):
        self.clean_dataset = clean_dataset
        self.indices = list(indices)
        self.sample_ids = [clean_dataset.sample_ids[i] for i in self.indices]
        self.preprocessor = preprocessor
        self.corruption = corruption

    def __len__(self):
        return len(self.indices)

    def _clean_sample(self, index):
        original_index = self.indices[index]
        sample = self.clean_dataset[original_index]
        if (sample.sample_id != self.clean_dataset.sample_ids[original_index]
                or sample.group_id != self.clean_dataset.group_ids[original_index]):
            raise ValueError("Clean sample metadata do not match dataset metadata")
        return sample

    def __getitem__(self, index):
        sample = self._clean_sample(index)
        x0 = self.preprocessor(sample.image)
        xT = self.corruption(x0.clone())
        validate_image(xT)
        if xT.shape != x0.shape:
            raise ValueError("Current bridge requires equal endpoint shapes")
        return {"x0": x0 * 2 - 1, "xT": xT * 2 - 1, "sample_id": sample.sample_id}
