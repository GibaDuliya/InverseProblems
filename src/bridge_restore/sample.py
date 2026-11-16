"""Reconstruct a fixed validation/test split using a trusted training checkpoint."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from .config import config_from_dict
from .checkpoint import load_checkpoint
from .data import build_data
from .bridge import VEBridge
from .models import build_model
from .sampling import SDESampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("batch-size and limit must be positive")
    state = load_checkpoint(args.checkpoint)
    config = config_from_dict(state["config"])
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Sampling output directory must be empty")
    # Official datasets read their assignments directly; custom partitions must exist.
    split_config = (config.split if config.split.mode == "official"
                    else replace(config.split, mode="load"))
    data = build_data(config.data, split_config, config.corruption, args.batch_size, config.seed)
    if data.split_manifest.checksum != state["split_checksum"] or data.corruption.to_config() != state["corruption_description"]:
        raise ValueError("Checkpoint data split/corruption mismatch")
    bridge = VEBridge(**asdict(config.bridge))
    model = build_model(config.model, bridge, config.data.image_size).to(args.device)
    model.load_state_dict(state["ema"]["parameters"] if args.weights == "ema" else state["model"])
    sampler_config = asdict(config.sampling)
    if args.steps is not None:
        sampler_config["num_steps"] = args.steps
    sampler = SDESampler(bridge, **sampler_config)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    loader = data.val_loader if args.split == "val" else data.test_loader
    output.mkdir(parents=True, exist_ok=True)
    (output / "sampling.json").write_text(json.dumps({**vars(args), "sampler": sampler_config,
                                                    "split_checksum": data.split_manifest.checksum}, indent=2))
    count = 0
    with (output / "index.jsonl").open("w") as index_file:
        for batch in loader:
            remaining = len(batch["sample_id"]) if args.limit is None else args.limit - count
            if remaining <= 0:
                break
            observation = batch["xT"][:remaining].to(args.device)
            result = sampler.sample(model, observation, generator)
            for key, image in zip(batch["sample_id"][:remaining], result.reconstruction.cpu()):
                filename = f"{count:08d}"
                # Preserve floating-point output, and save a clipped PNG for viewing.
                np.save(output / f"{filename}.npy", image.numpy())
                array = ((image.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
                Image.fromarray(array).save(output / f"{filename}.png")
                index_file.write(json.dumps({"sample_id": key, "file": filename, "nfe": result.nfe}) + "\n")
                count += 1
    print(f"Saved {count} reconstructions to {output}")


if __name__ == "__main__":
    main()
