"""Evaluate a trusted checkpoint with the same evaluator used during training."""
import argparse
import random
from dataclasses import asdict, replace
import numpy as np
import torch
import yaml
from .checkpoint import load_checkpoint
from .config import config_from_dict
from .data import build_data
from .bridge import VEBridge
from .models import build_model
from .sampling import SDESampler
from .evaluation import EvaluationConfig, RestorationEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    state = load_checkpoint(args.checkpoint)
    cfg = config_from_dict(state['config'])
    with open(args.config) as f: ec = EvaluationConfig(**yaml.safe_load(f))
    split = cfg.split if cfg.split.mode == 'official' else replace(cfg.split, mode='load')
    data = build_data(cfg.data, split, cfg.corruption, ec.batch_size, cfg.seed)
    if data.split_manifest.checksum != state['split_checksum'] or data.corruption.to_config() != state['corruption_description']:
        raise ValueError('Checkpoint data mismatch')
    bridge = VEBridge(**asdict(cfg.bridge))
    model = build_model(cfg.model, bridge, cfg.data.image_size).to(args.device)
    model.load_state_dict(state['ema']['parameters'] if ec.weights == 'ema' else state['model'])
    if args.split == 'test':
        ec = replace(ec, visual_samples_path=ec.test_visual_samples_path or ec.visual_samples_path,
                     metrics=ec.test_metrics or ec.metrics, criteria={},
                     selection_metric=next(iter(ec.test_metrics or ec.metrics)))
    loader = data.val_loader if args.split == 'val' else data.test_loader
    evaluator = RestorationEvaluator(loader.dataset, data.corruption, SDESampler(bridge, **asdict(cfg.sampling)), ec, args.device, split=args.split)
    report = evaluator.evaluate(model)
    report.protocol.update({'checkpoint': args.checkpoint, 'checkpoint_step': state['step'], 'split': args.split,
                            'split_checksum': data.split_manifest.checksum, 'seed': args.seed, 'experiment': asdict(cfg)})
    report.save(args.output)

if __name__ == '__main__': main()
