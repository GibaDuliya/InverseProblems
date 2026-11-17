from types import SimpleNamespace
import json
from bridge_restore import sweep
from bridge_restore.evaluation import EvaluationConfig


def test_single_epoch_no_validation_final_test(tiny_config, loaders, monkeypatch):
    tiny_config.model.mode = 'bm'
    tiny_config.training.grad_accum_steps = 4
    tiny_config.evaluation = EvaluationConfig(enabled=False, subset_size=None, test_subset_size=None)
    bundle = SimpleNamespace(train_loader=loaders[0],val_loader=loaders[1],test_loader=loaders[1],
        corruption=SimpleNamespace(to_config=lambda: {}),split_manifest=SimpleNamespace(checksum='split'))
    monkeypatch.setattr(sweep,'build_data',lambda *a: bundle)
    monkeypatch.setattr(sweep,'save_manifest',lambda *a: None)
    monkeypatch.setattr(sweep.Trainer,'validate',lambda *a: (_ for _ in ()).throw(AssertionError('No validation')))
    result=sweep.run_experiment(tiny_config,epochs=1)
    from pathlib import Path
    output=Path(tiny_config.output_dir)
    assert result['steps'] == 1
    assert (output/'training_loss.png').exists()
    assert (output/'test/summary.json').exists()
    assert not (output/'checkpoints/best.pt').exists()
    assert json.loads((output/'test/protocol.json').read_text())['split']=='test'


def test_sweep_reproduces_training_and_sampling(tiny_config, loaders, monkeypatch, tmp_path):
    import torch
    from pathlib import Path
    from bridge_restore.checkpoint import load_checkpoint
    tiny_config.model.mode = 'bm'
    tiny_config.evaluation = EvaluationConfig(enabled=False, subset_size=None, test_subset_size=None)
    bundle = SimpleNamespace(train_loader=loaders[0],val_loader=loaders[1],test_loader=loaders[1],
        corruption=SimpleNamespace(to_config=lambda: {}),split_manifest=SimpleNamespace(checksum='split'))
    monkeypatch.setattr(sweep,'build_data',lambda *a: bundle)
    monkeypatch.setattr(sweep,'save_manifest',lambda *a: None)
    for name in ['first', 'second']:
        tiny_config.output_dir = str(tmp_path/name)
        sweep.run_experiment(tiny_config, epochs=1, max_updates=2)
    assert (tmp_path/'first/metrics.jsonl').read_text() == (tmp_path/'second/metrics.jsonl').read_text()
    assert (tmp_path/'first/test/per_image.jsonl').read_text() == (tmp_path/'second/test/per_image.jsonl').read_text()
    a,b = [load_checkpoint(tmp_path/n/'checkpoints/last.pt') for n in ['first','second']]
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
