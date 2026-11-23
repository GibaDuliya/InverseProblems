"""JPEG observations and quantization metadata from the same encoded stream."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from bridge_restore.data.build import build_data
from bridge_restore.data.corruptions import JPEGCorruption
from bridge_restore.data.datasets import RestorationDataset


def read_jpeg_metadata(encoded, subsampling):
    try:
        import jpeglib
    except ImportError as exc:
        raise RuntimeError("Install the soc-jpeg extra to read JPEG coefficients") from exc

    with TemporaryDirectory(prefix="bridge-soc-jpeg-") as directory:
        path = Path(directory) / "observation.jpg"
        path.write_bytes(encoded)
        jpeg = jpeglib.read_dct(str(path))
        # jpeglib reports (vertical, horizontal) sampling factors.
        chroma_stride = {0: (1, 1), 1: (1, 2), 2: (2, 2)}[subsampling]
        if np.asarray(jpeg.samp_factor).tolist() != [list(chroma_stride), [1, 1], [1, 1]]:
            raise ValueError("Unexpected JPEG component sampling factors")
        return tuple(
            {"indices": torch.from_numpy(coefficients.astype(np.int32, copy=True)),
             "table": torch.from_numpy(jpeg.qt[table_id].astype(np.float32, copy=True))}
            for coefficients, table_id in zip((jpeg.Y, jpeg.Cb, jpeg.Cr), jpeg.quant_tbl_no)
        )


class JPEGRestorationDataset(RestorationDataset):
    def __init__(self, clean_dataset, indices, preprocessor, corruption):
        if not isinstance(corruption, JPEGCorruption):
            raise ValueError("jpeg_consistency requires JPEGCorruption")
        super().__init__(clean_dataset, indices, preprocessor, corruption)

    def __getitem__(self, index):
        sample = self._clean_sample(index)
        x0 = self.preprocessor(sample.image)
        encoded = self.corruption.encode(x0)
        observation = self.corruption.decode(encoded)
        return {"x0": x0 * 2 - 1, "xT": observation * 2 - 1, "sample_id": sample.sample_id,
                "jpeg": read_jpeg_metadata(encoded, self.corruption.subsampling)}


def build_soc_data(config, batch_size, seed, reward_name):
    factory = JPEGRestorationDataset if reward_name == "jpeg_consistency" else RestorationDataset
    return build_data(config.data, config.split, config.corruption, batch_size, seed,
                      dataset_factory=factory)
