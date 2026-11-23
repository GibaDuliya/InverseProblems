"""Train a VE-bridge denoiser with Adjoint Matching."""
import argparse
from dataclasses import asdict, replace
import hashlib
from math import ceil
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from bridge_restore.checkpoint import load_checkpoint
from bridge_restore.config import config_from_dict
from bridge_restore.data.splits import save_manifest

from .config import load_soc_config
from .data import build_soc_data
from .evaluation import resolve_evaluation
from .initialization import initialize_control_denoiser, teacher_from_state
from .rewards.jpeg import JPEGConsistencyReward
from .sampling import SOCSampler
from .training import SOCTrainer


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    config = load_soc_config(args.config)
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()) and args.resume is None:
        raise ValueError("Output directory is nonempty; use a new directory or --resume")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    state = load_checkpoint(config.control.teacher_checkpoint)
    if state.get("kind") == "bridge_soc":
        raise ValueError("Expected a base BM/ABM teacher checkpoint")
    base = config_from_dict(state["config"])
    if base.split.mode != "official":
        base.split = replace(base.split, mode="load")
    data = build_soc_data(base, config.training.batch_size, config.seed, config.reward.name)
    if (data.split_manifest.checksum != state["split_checksum"]
            or data.corruption.to_config() != state["corruption_description"]):
        raise ValueError("Teacher checkpoint does not match the split or corruption")
    if not len(data.train_loader) or not len(data.val_loader):
        raise ValueError("SOC requires nonempty training and validation splits")

    steps_per_pass = ceil(len(data.train_loader) / config.training.grad_accum_steps)
    budget = config.training.max_steps or steps_per_pass
    config = replace(config, training=replace(config.training, max_steps=budget),
                     evaluation=resolve_evaluation(base, config))
    print(f"train_images={len(data.train_loader.dataset)} steps_per_pass={steps_per_pass} "
          f"max_steps={budget} partial_pass={budget < steps_per_pass}", flush=True)

    reference = {"sha256": _file_digest(config.control.teacher_checkpoint),
                 "weights": config.control.teacher_weights}
    teacher = teacher_from_state(state, config.control.teacher_weights, config.training.device)
    del state
    student = initialize_control_denoiser(teacher, config.control)
    sampling = asdict(base.sampling)
    sampling.update({key: value for key, value in asdict(config.sampling).items() if value is not None})
    config = replace(config, sampling=replace(config.sampling, **sampling))
    sampler = SOCSampler(teacher.bridge, **sampling)
    reward = JPEGConsistencyReward(data.corruption.subsampling).to(config.training.device)
    provenance = {
        "teacher": reference, "architecture": asdict(replace(base.model, mode=student.mode)),
        "image_size": base.data.image_size, "bridge": asdict(base.bridge), "sampling": sampling,
        "split_checksum": data.split_manifest.checksum, "corruption": data.corruption.to_config(),
        "data": asdict(base.data),
    }
    trainer = SOCTrainer(teacher, student, sampler, reward, data, config, provenance)
    if args.resume is not None:
        trainer.resume(args.resume)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    save_manifest(data.split_manifest, output / "split.json")
    trainer.fit()


if __name__ == "__main__":
    main()
