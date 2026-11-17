from dataclasses import asdict
from types import SimpleNamespace
import math
import json
import pytest
import torch
from bridge_restore.evaluation.selection import BestSelection
from bridge_restore.evaluation import EvaluationConfig
from bridge_restore.evaluation.report import EvaluationReport
from bridge_restore.bridge import VEBridge
from bridge_restore.models import build_model
from bridge_restore.training import Trainer
from bridge_restore.checkpoint import load_checkpoint
from bridge_restore.config import config_from_dict
from bridge_restore.grid import configurations


def test_independent_selection_ties_and_nonfinite():
    selector = BestSelection({'a': {'metric':'psnr', 'mode':'max', 'test':True},
                              'b': {'metric':'lpips','mode':'min','test':False}})
    def update(p,l,step): return selector.update({'restoration': {'psnr':{'mean':p},'lpips':{'mean':l}}},step,step)
    assert update(10,.5,1) == ['a','b']
    assert update(9,.4,2) == ['b']
    assert update(10,.4,3) == []
    assert selector.best['a']['step'] == 1
    with pytest.raises(FloatingPointError): update(20,math.nan,4)
    assert selector.best['a']['step'] == 1  # atomic comparison, no partial changes


@pytest.mark.parametrize('name', ['../bad','a/b',''])
def test_invalid_criterion_names(name):
    with pytest.raises(ValueError):
        EvaluationConfig(criteria={name: {'metric':'psnr','mode':'max','test':True}})


@pytest.mark.parametrize('weights',['ema','raw'])
def test_epoch_fit_multicriteria_and_test_flags(tiny_config, loaders, monkeypatch, weights):
    import bridge_restore.training as module
    calls=[]
    class Evaluator:
        def __init__(self, dataset, corruption, sampler, config, device, split='val'):
            self.split=split
            self.n=0
        def evaluate(self, model):
            self.n+=1
            calls.append(self.split)
            return EvaluationReport({'restoration':{'psnr':{'mean':10-self.n},'mae':{'mean':1/self.n}}},
                [{'sample_id':'x'}], {'config':{'weights':weights}}, {})
    monkeypatch.setattr(module,'RestorationEvaluator',Evaluator)
    tiny_config.training.max_steps=None
    tiny_config.training.max_epochs=2
    tiny_config.training.grad_accum_steps=2
    tiny_config.evaluation=EvaluationConfig(enabled=True,weights=weights,unit='epoch',interval=1,
        metrics={'psnr':{},'mae':{}},criteria={
        'p':{'metric':'psnr','mode':'max','test':True},
        'm':{'metric':'mae','mode':'min','test':True},
        'skip':{'metric':'mae','mode':'min','test':False}})
    bridge=VEBridge(**asdict(tiny_config.bridge))
    model=build_model(tiny_config.model,bridge,8)
    trainer=Trainer(model,bridge,*loaders[:2],tiny_config,'split',{},SimpleNamespace(corruption=None,test_loader=loaders[2]))
    trainer.fit()
    assert trainer.completed_epochs==2 and trainer.step==4
    assert trainer.samples_seen==12
    assert calls==['val','val','test','test']
    assert not (trainer.output_dir/'checkpoints').exists()
    assert (trainer.output_dir/'validation/epoch_0002/summary.json').exists()
    assert not (trainer.output_dir/'test/skip').exists()
    assert trainer.model.training
    assert load_checkpoint(trainer.output_dir/'best/p/checkpoint.pt')['step']==2
    assert load_checkpoint(trainer.output_dir/'best/m/checkpoint.pt')['step']==4
    assert json.loads((trainer.output_dir/'best/p/protocol.json').read_text())['step']==2


def test_identical_test_checkpoints_reuse_generation(tiny_config, loaders, monkeypatch):
    import bridge_restore.training as module
    calls=[]
    class Evaluator:
        def __init__(self,*args,split='val',**kwargs): self.split=split
        def evaluate(self,model):
            calls.append(self.split)
            return EvaluationReport({'restoration':{'psnr':{'mean':10}}},[],{})
    monkeypatch.setattr(module,'RestorationEvaluator',Evaluator)
    tiny_config.training.max_steps=1
    tiny_config.evaluation=EvaluationConfig(enabled=True,criteria={
        name:{'metric':'psnr','mode':'max','test':True} for name in ['a','b']})
    bridge=VEBridge(**asdict(tiny_config.bridge))
    trainer=Trainer(build_model(tiny_config.model,bridge,8),bridge,*loaders[:2],tiny_config,'split',{},
                    SimpleNamespace(corruption=None,test_loader=loaders[2]))
    trainer.fit()
    assert calls==['val','test']
    assert (trainer.output_dir/'test/a/summary.json').read_text()==(trainer.output_dir/'test/b/summary.json').read_text()


def test_epoch_config_and_grid():
    import yaml
    from pathlib import Path
    config_from_dict({'training':{'max_epochs':2,'max_steps':None}})
    with pytest.raises(ValueError): config_from_dict({'training':{'max_epochs':None,'max_steps':None}})
    runs=list(configurations(yaml.safe_load(Path('configs/celeba_main_grid.yaml').read_text())))
    assert len(runs)==12
    assert len({cfg.output_dir for _,cfg in runs})==12
    assert all(cfg.training.max_steps is None and cfg.training.max_epochs==20 for _,cfg in runs)
    assert all(cfg.training.batch_size*cfg.training.grad_accum_steps==64 for _,cfg in runs)
