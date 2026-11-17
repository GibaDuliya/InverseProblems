from dataclasses import asdict
from types import SimpleNamespace
import pytest
from bridge_restore.bridge import VEBridge
from bridge_restore.models import build_model
from bridge_restore.training import Trainer
from bridge_restore.evaluation import EvaluationConfig

@pytest.mark.parametrize('unit', ['step', 'epoch'])
@pytest.mark.parametrize('weights', ['raw', 'ema'])
def test_fit_selects_best_and_tests(tiny_config, loaders, tmp_path, unit, weights):
    tiny_config.evaluation = EvaluationConfig(enabled=True, weights=weights, unit=unit,
        interval=2, subset_size=None, final_test=True)
    tiny_config.training.max_steps = 3
    bridge = VEBridge(**asdict(tiny_config.bridge))
    model = build_model(tiny_config.model, bridge, 8)
    data = SimpleNamespace(corruption=None, test_loader=loaders[1])
    trainer = Trainer(model, bridge, loaders[0], loaders[1], tiny_config, 'split', {}, data)
    trainer.fit()
    assert trainer.last_validation_step == 3
    assert (trainer.output_dir/'best/psnr/checkpoint.pt').exists()
    assert (trainer.output_dir/'test/psnr/summary.json').exists()
    assert trainer.model.training


def test_visual_panel_included_ordered_and_saved(tmp_path):
    import json
    import torch
    from PIL import Image
    from bridge_restore.evaluation import RestorationEvaluator
    class Pairs(torch.utils.data.Dataset):
        sample_ids = ['a', 'b', 'c', 'd']
        def __len__(self): return 4
        def __getitem__(self, i):
            x = torch.full((3,8,8), i/4)
            return {'x0': x, 'xT': x, 'sample_id': self.sample_ids[i]}
    class Sampler:
        num_steps=1; start_offset=.001; clip_output=True
        calls=0
        def sample(self, model, observation):
            self.calls += 1
            return SimpleNamespace(reconstruction=observation, nfe=2)
    path = tmp_path/'ids.json'
    path.write_text(json.dumps({'split':'val','sample_ids':['d','b']}))
    config = EvaluationConfig(subset_size=3, batch_size=2, visual_samples_path=str(path))
    sampler = Sampler()
    evaluator = RestorationEvaluator(Pairs(), None, sampler, config, 'cpu')
    report = evaluator.evaluate(torch.nn.Identity())
    assert list(report.visual_examples) == ['d','b']
    assert {'b','d'} <= set(report.protocol['sample_ids'])
    assert len(report.per_image) == 3 and sampler.calls == 2
    report.save(tmp_path/'report')
    with Image.open(tmp_path/'report/images/panel_00.png') as image:
        assert image.size == (24, 88)
    with pytest.raises(ValueError, match='split mismatch'):
        RestorationEvaluator(Pairs(), None, sampler, config, 'cpu', split='test')
    config.subset_size = 1
    with pytest.raises(ValueError, match='smaller'):
        RestorationEvaluator(Pairs(), None, sampler, config, 'cpu')
    path.write_text(json.dumps({'split':'val','sample_ids':['missing']}))
    config.subset_size = 3
    with pytest.raises(ValueError, match='absent'):
        RestorationEvaluator(Pairs(), None, sampler, config, 'cpu')


def test_celeba_visual_ids_distinct_and_correct_split():
    import json
    from pathlib import Path
    if not Path('data/celeba/identity_CelebA.txt').exists():
        pytest.skip('Local CelebA metadata unavailable')
    seen = set()
    partitions = dict(line.split() for line in Path('data/celeba/list_eval_partition.txt').read_text().splitlines())
    identities = dict(line.split() for line in Path('data/celeba/identity_CelebA.txt').read_text().splitlines())
    for split, code, size in [('val','1',16),('test','2',32)]:
        value = json.loads(Path(f'configs/visual_samples/celeba_{split}.json').read_text())
        assert len(value['sample_ids']) == size
        for key in value['sample_ids']:
            assert partitions[key] == code
            assert identities[key] not in seen
            seen.add(identities[key])
