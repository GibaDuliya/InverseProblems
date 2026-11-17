import torch
from bridge_restore.data.datasets import RestorationDataset, ImagePreprocessor, CelebADataset
from bridge_restore.data.corruptions import JPEGCorruption
from PIL import Image
import numpy as np


def test_jpeg_pair_order_and_immutability(clean):
    corruption = JPEGCorruption(quality=10, subsampling=2)
    preprocessing = ImagePreprocessor(8)
    original = clean[0].image.clone()
    dataset = RestorationDataset(clean, [0], preprocessing, corruption)
    pair = dataset[0]
    expected_clean = preprocessing(original)
    expected_degraded = corruption(expected_clean)
    assert torch.equal(clean[0].image, original)
    assert torch.equal(pair["x0"], expected_clean * 2 - 1)
    assert torch.equal(pair["xT"], expected_degraded * 2 - 1)
    assert torch.equal(pair["xT"], dataset[0]["xT"])
    assert pair["sample_id"] == clean.sample_ids[0]
    assert pair["xT"].shape == (3, 8, 8)
    assert not torch.equal(pair["x0"], pair["xT"])
    assert pair["xT"].min() >= -1 and pair["xT"].max() <= 1


def test_celeba_adapter(tmp_path):
    (tmp_path / "img_align_celeba").mkdir()
    for i in range(3):
        Image.fromarray(np.full((8, 8, 3), 40*i, dtype=np.uint8)).save(tmp_path / "img_align_celeba" / f"{i}.jpg")
    (tmp_path / "identity_CelebA.txt").write_text("0.jpg 1\n1.jpg 2\n2.jpg 3\n")
    (tmp_path / "list_eval_partition.txt").write_text("0.jpg 0\n1.jpg 1\n2.jpg 2\n")
    dataset = CelebADataset(tmp_path)
    assert dataset[1].group_id is None
    assert dataset.official_split_assignments["2.jpg"] == "test"
    assert CelebADataset(tmp_path, identity_file="identity_CelebA.txt")[1].group_id == "2"
    (tmp_path / "identity_CelebA.txt").unlink()
    assert len(CelebADataset(tmp_path)) == 3
