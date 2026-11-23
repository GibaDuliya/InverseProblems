"""Assemble a DataBundle; training only consumes its paired loaders."""
from dataclasses import dataclass
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from .datasets import ImagePreprocessor, RestorationDataset
from .corruptions import Corruption, build_corruption, resolve_factory
from .splits import SplitManifest, load_or_create_split


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    corruption: Corruption
    split_manifest: SplitManifest


class EpochSampler(Sampler):
    """Permutation determined only by seed and epoch, not global torch RNG."""
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self):
        return len(self.dataset)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_clean_dataset(config):
    return resolve_factory(config.factory)(**config.kwargs)


def build_loaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers, seed):
    loaders = []
    for index, dataset in enumerate((train_dataset, val_dataset, test_dataset)):
        loaders.append(DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            sampler=EpochSampler(dataset, seed) if index == 0 else None,
            shuffle=False, drop_last=False, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed + index),
            pin_memory=torch.cuda.is_available()))
    return loaders


def build_data(data_config, split_config, corruption_config, batch_size, seed, *, dataset_factory=RestorationDataset):
    clean = build_clean_dataset(data_config)
    manifest = load_or_create_split(clean, split_config)
    corruption = build_corruption(corruption_config)
    preprocessing = ImagePreprocessor(data_config.image_size, data_config.crop_mode,
                                     data_config.interpolation, data_config.antialias)
    lookup = {key: index for index, key in enumerate(clean.sample_ids)}
    datasets = [dataset_factory(clean, [lookup[key] for key in ids], preprocessing, corruption)
                for ids in (manifest.train_ids, manifest.val_ids, manifest.test_ids)]
    loaders = build_loaders(*datasets, batch_size, data_config.num_workers, seed)
    return DataBundle(*loaders, corruption, manifest)
