"""Independent BM training runs, no validation; evaluate final weights on fixed IDs."""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import random
from dataclasses import asdict, replace
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from .config import config_from_dict, save_config
from .data import build_data
from .data.splits import save_manifest
from .bridge import VEBridge
from .models import build_model
from .training import Trainer
from .sampling import SDESampler
from .evaluation import RestorationEvaluator


def plot_loss(records, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values = [r for r in records if r['updated']]
    if not values: raise RuntimeError('No successful optimizer updates')
    steps = [r['step'] for r in values]
    loss = np.array([r['loss'] for r in values])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, loss, alpha=.25, linewidth=.6, label='training loss')
    window = min(100, len(loss))
    ax.plot(steps[window-1:], np.convolve(loss, np.ones(window)/window, mode='valid'), label=f'moving mean ({window} updates)')
    ax.set(xlabel='Successful optimizer updates', ylabel='Weighted denoising loss')
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run_experiment(config, epochs=1, max_updates=None):
    """Fresh initialization per T. Epochs count data passes, including AMP skips."""
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()): raise ValueError(f'Nonempty output: {output}')
    if config.evaluation.enabled: raise ValueError('Sweep requires evaluation.enabled=false (no validation)')
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    data = build_data(config.data, config.split, config.corruption, config.training.batch_size, config.seed)
    bridge = VEBridge(**asdict(config.bridge))
    model = build_model(config.model, bridge, config.data.image_size)
    trainer = Trainer(model, bridge, data.train_loader, data.val_loader, config,
                      data.split_manifest.checksum, data.corruption.to_config())
    # This loop deliberately does not call fit(): no validation or best-checkpoint selection.
    updates_per_epoch = math.ceil(len(data.train_loader)/config.training.grad_accum_steps)
    target = epochs*updates_per_epoch
    if max_updates is not None: target = min(target, max_updates)
    config.training.max_steps = target
    save_config(config, output/'config.yaml')
    save_manifest(data.split_manifest, output/'split.json')
    records = []
    skipped = 0
    with (output/'metrics.jsonl').open('w') as log, tqdm(total=target, desc=f'T={bridge.sigma_max:g}') as bar:
        while trainer.step < target:
            values = trainer.train_step()
            skipped = 0 if values['updated'] else skipped+1
            if skipped >= 20: raise FloatingPointError('20 consecutive AMP overflows')
            row = {'step':trainer.step, 'epoch':trainer.data_position['epoch'], **values}
            log.write(json.dumps(row)+'\n'); log.flush(); records.append(row)
            if values['updated']:
                bar.update(1); bar.set_postfix(loss=f"{values['loss']:.5g}")
                if trainer.step % config.training.checkpoint_interval == 0:
                    trainer.save(output/'checkpoints'/f'step_{trainer.step:08d}.pt')
            if trainer.data_position['epoch']+1 >= epochs and trainer.data_position['batch'] == len(data.train_loader): break
    trainer.save(output/'checkpoints/last.pt')
    plot_loss(records, output/'training_loss.png')
    ec = replace(config.evaluation, visual_samples_path=config.evaluation.test_visual_samples_path,
                 subset_size=config.evaluation.test_subset_size,
                 metrics=config.evaluation.test_metrics or config.evaluation.metrics)
    evaluator = RestorationEvaluator(data.test_loader.dataset, data.corruption,
        SDESampler(bridge, **asdict(config.sampling)), ec, trainer.device, split='test')
    inference_model = copy.deepcopy(trainer.model).eval().requires_grad_(False) if ec.weights == 'ema' else trainer.model
    if ec.weights == 'ema': trainer.ema.copy_to(inference_model)
    report = evaluator.evaluate(inference_model)
    report.protocol.update({'checkpoint_step':trainer.step, 'split':'test', 'experiment':asdict(config),
                            'split_checksum':data.split_manifest.checksum, 'checkpoint':str(output/'checkpoints/last.pt')})
    report.save(output/'test')
    return {'T':bridge.sigma_max, 'steps':trainer.step, 'samples_seen':trainer.samples_seen, 'test':report.summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--T', type=float, help='Run only this configured T')
    args = parser.parse_args()
    with open(args.config) as f: sweep = yaml.safe_load(f)
    if set(sweep)-{'times','epochs','max_updates','experiment'}: raise ValueError('Unknown sweep fields')
    epochs = sweep.get('epochs',1); cap = sweep.get('max_updates')
    if not isinstance(epochs,int) or epochs < 1 or (cap is not None and (not isinstance(cap,int) or cap < 1)):
        raise ValueError('epochs/max_updates must be positive integers')
    times = sweep['times']
    if not times or len(set(times)) != len(times) or any(not math.isfinite(t) or t <= 0 for t in times): raise ValueError('Invalid times')
    if args.T is not None:
        if args.T not in times: raise ValueError('--T must be in configured times')
        times = [args.T]
    results=[]
    for time in times:
        values=copy.deepcopy(sweep['experiment'])
        values['bridge']['sigma_max']=time
        values['output_dir']=str(Path(values['output_dir'])/f'T_{time:g}')
        result=run_experiment(config_from_dict(values), epochs, cap)
        results.append(result)
        Path(values['output_dir'],'result.json').write_text(json.dumps(result,indent=2))
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(json.dumps(results,indent=2))

if __name__ == '__main__': main()
