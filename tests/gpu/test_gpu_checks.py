"""Opt-in pytest diagnostics for a small CUDA card."""
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import Subset
from bridge_restore.config import load_config, save_config
from bridge_restore.bridge import VEBridge
from bridge_restore.models import build_model
from bridge_restore.sampling import SDESampler
from bridge_restore.checkpoint import load_checkpoint
from bridge_restore.data import build_data
from bridge_restore.data.build import build_loaders
from bridge_restore.metrics import MetricBatch, build_metrics
from bridge_restore.evaluation import RestorationEvaluator
from bridge_restore.training import Trainer


def sync(device):
    if device == 'cuda': torch.cuda.synchronize()


def timed(call, device):
    sync(device)
    start=time.perf_counter()
    result=call()
    sync(device)
    return result, time.perf_counter()-start


def finite_summary(report):
    for group in ('restoration','input'):
        for name, result in report.summary[group].items():
            if not math.isfinite(result['mean']):
                raise AssertionError(f'Nonfinite {group}/{name}: {result}')


import pytest
from types import SimpleNamespace

pytestmark = pytest.mark.gpu

@pytest.fixture
def gpu_case(request, gpu_output):
    if not request.config.getoption("--run-gpu"):
        pytest.skip("Enable explicitly with --run-gpu")
    args = SimpleNamespace(
        config=request.config.getoption("--gpu-config"),
        device=request.config.getoption("--gpu-device"),
        batch_size=request.config.getoption("--gpu-batch-size"),
        samples=request.config.getoption("--gpu-samples"),
        checkpoint=request.config.getoption("--gpu-checkpoint"),
        mode=request.node.name,
    )
    if args.samples < 2 or args.batch_size < 1:
        pytest.fail("gpu-samples >= 2 and gpu-batch-size >= 1 required")
    if args.device == "cuda" and not torch.cuda.is_available():
        pytest.fail("CUDA unavailable in this PyTorch environment")
    output = gpu_output / request.node.name
    if output.exists() and any(output.iterdir()):
        pytest.fail(f"Use a new --gpu-output: {output} already contains results")
    output.mkdir(parents=True, exist_ok=True)
    cfg=load_config(args.config)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    cfg.training.device=args.device
    cfg.training.precision='float32'
    cfg.training.batch_size=args.batch_size
    cfg.data.num_workers=0
    cfg.output_dir=str(output/'training')
    data=build_data(cfg.data,cfg.split,cfg.corruption,args.batch_size,cfg.seed)
    metric_config=copy.deepcopy(cfg.evaluation.test_metrics or cfg.evaluation.metrics)
    original_kid=copy.deepcopy(metric_config.get('kid'))
    if 'kid' in metric_config:
        metric_config['kid']['subset_size']=min(args.samples,metric_config['kid'].get('subset_size',1000))
    info={'mode':args.mode,'torch':torch.__version__,'cuda':torch.version.cuda,
        'device':torch.cuda.get_device_name(0) if args.device=='cuda' else 'cpu',
        'precision':'float32','batch_size':args.batch_size,'metrics':metric_config,
        'original_kid':original_kid,'checkpoint':args.checkpoint,'experiment':asdict(cfg)}
    if args.device=='cuda': torch.cuda.reset_peak_memory_stats()
    yield args, cfg, data, metric_config, info, output
    report = getattr(request.node, "call_report", None)
    info["status"] = "passed" if report and report.passed else "failed"
    if report and report.failed:
        info["error"] = str(report.longrepr)
    if args.device == "cuda":
        info["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        info["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    (output / "diagnostics.json").write_text(json.dumps(info, indent=2, allow_nan=False))
    print(f"Report: {output / 'diagnostics.json'}")

def test_metrics(gpu_case):
    args, cfg, data, metric_config, info, output = gpu_case
    # Select genuine test images; JPEG predictions avoid infinite identity PSNR.
    dataset=data.test_loader.dataset
    indices=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(cfg.seed))[:args.samples].tolist()
    batches=torch.utils.data.DataLoader(Subset(dataset,indices),batch_size=args.batch_size)
    results={}
    for name, params in metric_config.items():
        metric,load_seconds=timed(lambda:build_metrics({name:params},args.device,data.corruption)[name],args.device)
        summaries=[]
        for repeat in range(2):
            metric.reset()
            for batch in batches:
                x0,xT=batch['x0'].to(args.device),batch['xT'].to(args.device)
                with torch.no_grad(): metric.update(MetricBatch(xT,x0,batch['sample_id'],xT))
            result=metric.compute()
            if not math.isfinite(result.summary['mean']): raise AssertionError(f'Nonfinite {name}')
            summaries.append(result.summary)
        if name!='kid' and not math.isclose(summaries[0]['mean'],summaries[1]['mean'],rel_tol=1e-5,abs_tol=1e-5):
            raise AssertionError(f'{name}: reset/repeat mismatch')
        if name in ('arcface','lpips','ssim','observation_mse','observation_mae'):
            batch=next(iter(batches))
            x0,xT=batch['x0'].to(args.device),batch['xT'].to(args.device)
            with torch.no_grad():
                identity=metric.values_for(MetricBatch(x0,x0,batch['sample_id'],xT))
            valid=identity[torch.isfinite(identity)]
            if not len(valid): raise AssertionError(f'{name}: no valid identity scores')
            expected=1. if name in ('arcface','ssim') else 0.
            torch.testing.assert_close(valid,torch.full_like(valid,expected),atol=1e-4,rtol=1e-4)
        results[name]={'load_seconds':load_seconds,'runs':summaries}
        print(name,results[name],flush=True)
        del metric
    # Exact arithmetic and identity checks independent of pretrained scores.
    from bridge_restore.metrics.pixel import PixelMetric
    for name,expected in [('mse',1.),('mae',1.),('psnr',0.)]:
        m=PixelMetric(name)
        value=m.values_for(MetricBatch(torch.ones(1,3,8,8),-torch.ones(1,3,8,8),['synthetic']))
        torch.testing.assert_close(value,torch.tensor([expected]))
    info['results']=results


def test_evaluation_timing(gpu_case):
    args, cfg, data, metric_config, info, output = gpu_case
    bridge=VEBridge(**asdict(cfg.bridge))
    model=build_model(cfg.model,bridge,cfg.data.image_size).to(args.device).eval()
    if args.checkpoint:
        state=load_checkpoint(args.checkpoint)
        for section in ('model','bridge','data','corruption'):
            current=asdict(getattr(cfg,section))
            old=copy.deepcopy(state['config'][section])
            if section=='data': current.pop('num_workers'); old.pop('num_workers')
            if current!=old: raise ValueError(f'Checkpoint {section} differs from config')
        model.load_state_dict(state['ema']['parameters'] if cfg.evaluation.weights=='ema' else state['model'])
    ec=replace(cfg.evaluation,subset_size=args.samples,batch_size=args.batch_size,
        visual_samples_path=cfg.evaluation.test_visual_samples_path,metrics=metric_config,
        criteria={},selection_metric=next(iter(metric_config)))
    sampler=SDESampler(bridge,**asdict(cfg.sampling))
    evaluator,load_seconds=timed(lambda:RestorationEvaluator(data.test_loader.dataset,data.corruption,sampler,ec,args.device,split='test'),args.device)
    first=next(iter(evaluator.loader))['xT'].to(args.device)
    _,warmup_seconds=timed(lambda:sampler.sample(model,first),args.device)
    # Final FID/KID computation is not proportional to sample count.
    finalize=[0.]
    for group in (evaluator.metrics,evaluator.baseline):
        for metric in group.values():
            original=metric.compute
            def wrapped(original=original):
                value,seconds=timed(original,args.device)
                finalize[0]+=seconds
                return value
            metric.compute=wrapped
    report,seconds=timed(lambda:evaluator.evaluate(model),args.device)
    finite_summary(report)
    _,save_seconds=timed(lambda:report.save(output/'evaluation'),args.device)
    per_image=(seconds-finalize[0])/args.samples
    info.update(samples=args.samples,load_seconds=load_seconds,warmup_seconds=warmup_seconds,
        evaluation_seconds=seconds,finalize_seconds=finalize[0],save_seconds=save_seconds,
        seconds_per_image_excluding_finalize=per_image,
        estimated_4096_seconds_excluding_finalize=per_image*4096,
        naive_4096_seconds=seconds*4096/args.samples,
        note='Estimate applies only to this GPU/batch/FP32. Final FID/KID costs do not scale linearly; KID subset was reduced. Random weights are a timing smoke test, not restoration quality.')


def test_training_cycle(gpu_case):
    args, cfg, data, metric_config, info, output = gpu_case
    # Two tiny epochs, full-size U-Net, short SDE; exercise every phase.
    cfg.training.max_epochs=2; cfg.training.max_steps=None
    cfg.training.grad_accum_steps=2
    cfg.training.log_interval=1
    cfg.sampling.num_steps=4
    cfg.evaluation=replace(cfg.evaluation,unit='epoch',interval=1,subset_size=16,
        batch_size=args.batch_size,test_subset_size=32,
        metrics={'psnr':{},'ssim':{}},test_metrics={'psnr':{},'ssim':{}},
        criteria={'psnr':{'metric':'psnr','mode':'max','test':True},
                  'ssim':{'metric':'ssim','mode':'max','test':False}})
    train=Subset(data.train_loader.dataset,list(range(5)))
    loaders=build_loaders(train,data.val_loader.dataset,data.test_loader.dataset,args.batch_size,0,cfg.seed)
    data.train_loader,data.val_loader,data.test_loader=loaders
    bridge=VEBridge(**asdict(cfg.bridge))
    model=build_model(cfg.model,bridge,cfg.data.image_size)
    trainer=Trainer(model,bridge,*loaders[:2],cfg,data.split_manifest.checksum,data.corruption.to_config(),data)
    save_config(cfg,output/'resolved_cycle.yaml')
    _,seconds=timed(trainer.fit,args.device)
    expected_steps=2*math.ceil(math.ceil(5/args.batch_size)/2)
    assert trainer.step==expected_steps and trainer.samples_seen==10 and trainer.completed_epochs==2
    root=Path(cfg.output_dir)
    for name in cfg.evaluation.criteria:
        state=load_checkpoint(root/'best'/name/'checkpoint.pt')
        assert state['step']>0 and state['optimizer']['state']
        assert list((root/'best'/name/'images').glob('*.png'))
    assert (root/'test/psnr/summary.json').exists()
    assert not (root/'test/ssim').exists()
    assert not (root/'checkpoints/last.pt').exists()
    for epoch in (1,2): assert (root/f'validation/epoch_{epoch:04d}/summary.json').exists()
    info.update(seconds=seconds,steps=trainer.step,samples_seen=trainer.samples_seen,
        note='Cycle uses 5 train images, 16 validation, 32 test, 4 SDE steps, PSNR/SSIM. Full metrics and 100-step timing are separate modes.')


