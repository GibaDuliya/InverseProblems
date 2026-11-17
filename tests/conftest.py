import torch
import pytest
from torch.utils.data import Dataset
from bridge_restore.config import ExperimentConfig, ModelConfig, TrainingConfig
from bridge_restore.data.datasets import CleanSample, RestorationDataset, ImagePreprocessor
from bridge_restore.data.corruptions import JPEGCorruption
from bridge_restore.data.build import build_loaders


torch.set_num_threads(1)


class TinyClean(Dataset):
    def __init__(self, n=12):
        self.sample_ids = [f"image-{i}" for i in range(n)]
        self.group_ids = [f"person-{i//2}" for i in range(n)]
        self.images = torch.rand(n, 3, 10, 12, generator=torch.Generator().manual_seed(13))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        return CleanSample(self.images[index], self.sample_ids[index], self.group_ids[index])


@pytest.fixture
def clean():
    return TinyClean()


@pytest.fixture
def tiny_config(tmp_path):
    config = ExperimentConfig(output_dir=str(tmp_path / "run"))
    config.data.image_size = 8
    config.model = ModelConfig(channels=32, channel_mult=[1, 2], num_res_blocks=1,
                               attention_resolutions=[4], num_heads=1, dropout=.1)
    config.training = TrainingConfig(device="cpu", batch_size=2, max_steps=2,
                                     validation_interval=1, validation_batches=1,
                                     checkpoint_interval=2, log_interval=1)
    config.bridge.sigma_min = .02
    config.sampling.num_steps = 4
    return config


@pytest.fixture
def loaders(clean):
    corruption = JPEGCorruption()
    datasets = [RestorationDataset(clean, ids, ImagePreprocessor(8), corruption)
                for ids in (range(6), range(6, 8), range(8, 12))]
    return build_loaders(*datasets, batch_size=2, num_workers=0, seed=42)


def pytest_addoption(parser):
    group = parser.getgroup("gpu diagnostics")
    group.addoption("--run-gpu", action="store_true", help="Run real-data GPU diagnostics")
    group.addoption("--gpu-config", default="configs/train.yaml")
    group.addoption("--gpu-device", choices=["cuda", "cpu"], default="cuda")
    group.addoption("--gpu-output", default=None)
    group.addoption("--gpu-batch-size", type=int, default=1)
    group.addoption("--gpu-samples", type=int, default=50)
    group.addoption("--gpu-checkpoint", default=None)
    group.addoption("--gpu-threads", type=int, default=4)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: opt-in diagnostics using real data and pretrained metrics")


