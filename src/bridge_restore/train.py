"""CLI entry point for BM/ABM training."""
import argparse
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch
from .config import load_config, save_config
from .data import build_data
from .data.splits import save_manifest
from .bridge import VEBridge
from .models import build_model
from .training import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output directory is nonempty; use a new directory or --resume")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    data = build_data(config.data, config.split, config.corruption,
                      config.training.batch_size, config.seed)
    bridge = VEBridge(**asdict(config.bridge))
    model = build_model(config.model, bridge, config.data.image_size)
    trainer = Trainer(model, bridge, data.train_loader, data.val_loader, config,
                      data.split_manifest.checksum, data.corruption.to_config(), data_bundle=data)
    if args.resume:
        trainer.resume(args.resume)
    save_config(config, output / "config.yaml")
    save_manifest(data.split_manifest, output / "split.json")
    print(f"mode={config.model.mode}; parameters={sum(p.numel() for p in model.parameters()):,}")
    trainer.fit()


if __name__ == "__main__":
    main()
