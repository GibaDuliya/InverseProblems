import math

from PIL import Image
import pytest
import torch
from torch.utils.data import default_collate

from bridge_restore.data.corruptions import JPEGCorruption
from bridge_restore.data.datasets import ImagePreprocessor, RestorationDataset
from bridge_soc.data import JPEGRestorationDataset
from bridge_soc.rewards.jpeg import JPEGConsistencyReward, block_dct, interval_error


def test_interval_gradient_and_dct_axes():
    coefficients = torch.tensor([-4., -1., 5.], requires_grad=True)
    loss = interval_error(coefficients, torch.zeros_like(coefficients), torch.tensor(4.))
    torch.testing.assert_close(loss, torch.tensor([4., 0., 9.]))
    loss.sum().backward()
    torch.testing.assert_close(coefficients.grad, torch.tensor([-4., 0., 6.]))

    basis = JPEGConsistencyReward(2).basis
    dc = block_dct(torch.ones(8, 8), basis).view(8, 8)
    expected = torch.zeros_like(dc)
    expected[0, 0] = 8
    torch.testing.assert_close(dc, expected, atol=1e-6, rtol=0)
    pattern = torch.cos(math.pi * (torch.arange(8) + .5) / 8).expand(8, 8)
    result = block_dct(pattern, basis).view(8, 8)
    assert result.abs().flatten().argmax().item() == 1


@pytest.mark.parametrize("subsampling,progressive", [(0, False), (1, True), (2, False)])
def test_jpeg_reward_on_actual_stream(clean, subsampling, progressive):
    pytest.importorskip("jpeglib")
    codec = JPEGCorruption(quality=10, subsampling=subsampling, progressive=progressive)
    preprocessing = ImagePreprocessor(17)
    samples = JPEGRestorationDataset(clean, [0, 1], preprocessing, codec)
    original = RestorationDataset(clean, [0, 1], preprocessing, codec)
    batch = default_collate([samples[0], samples[1]])
    torch.testing.assert_close(batch["xT"], torch.stack([original[0]["xT"], original[1]["xT"]]))
    from io import BytesIO
    with Image.open(BytesIO(codec.encode(preprocessing(clean[0].image)))) as image:
        for metadata, (component, h, v, table_id) in zip(batch["jpeg"], image.layer):
            expected = torch.tensor(image.quantization[table_id]).reshape(8, 8).float()
            torch.testing.assert_close(metadata["table"][0], expected)

    reward = JPEGConsistencyReward(subsampling)
    # The continuous color conversion and DCT need not match codec rounding.
    clean_values = reward(batch["x0"], batch)
    candidate = (batch["x0"] + .5).requires_grad_(True)
    values = reward(candidate, batch)
    assert values.shape == (2,)
    assert torch.isfinite(values).all()
    assert (values < clean_values).all()
    values.sum().backward()
    assert candidate.grad.abs().sum() > 0
