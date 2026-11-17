import numpy as np
import pytest
import torch
from bridge_restore.metrics.base import MetricBatch
from bridge_restore.metrics.identity import ArcFaceMetric


def test_identity_missing_coverage_and_reset():
    metric=ArcFaceMetric.__new__(ArcFaceMetric)
    metric.min_coverage=.5
    metric.references={'a':None}
    metric.values={'a':float('nan'),'b':.8}
    result=metric.compute()
    assert result.summary['coverage']==.5
    assert result.per_image['a'] is None
    assert result.summary['missing_sample_ids']==['a']
    metric.min_coverage=.9
    with pytest.raises(ValueError,match='coverage'): metric.compute()
    metric.reset()
    assert metric.values=={} and metric.references=={'a':None}


def test_identity_rgb_to_bgr():
    red=torch.tensor([1.,-1.,-1.]).reshape(1,3,1,1)
    np.testing.assert_array_equal(ArcFaceMetric._bgr(red),np.array([[[[0,0,255]]]],dtype=np.uint8))


def test_real_quality_evaluator_when_requested(tmp_path):
    import os
    if os.environ.get('BRIDGE_QUALITY_SMOKE') != '1':
        pytest.skip('Opt-in: requires real pretrained weights and local CelebA')
    from pathlib import Path
    from types import SimpleNamespace
    from bridge_restore.data.datasets import read_rgb, ImagePreprocessor
    from bridge_restore.data.corruptions import JPEGCorruption
    from bridge_restore.evaluation import EvaluationConfig, RestorationEvaluator
    root=Path(os.environ['CELEBA_ROOT'])
    class Pairs(torch.utils.data.Dataset):
        sample_ids=['185390.jpg','186520.jpg']
        def __len__(self): return 2
        def __getitem__(self,i):
            x=ImagePreprocessor(128)(read_rgb(root/'img_align_celeba'/self.sample_ids[i]))
            return {'x0':2*x-1,'xT':2*JPEGCorruption()(x)-1,'sample_id':self.sample_ids[i]}
    class Sampler:
        num_steps=1;start_offset=.001;clip_output=True
        def sample(self,model,x):return SimpleNamespace(reconstruction=x,nfe=0)
    config=EvaluationConfig(subset_size=2,batch_size=2,metrics={
        'arcface':{'root':os.environ['INSIGHTFACE_ROOT']},'musiq':{}},
        criteria={'id':{'metric':'arcface','mode':'max','test':False}})
    evaluator=RestorationEvaluator(Pairs(),JPEGCorruption(),Sampler(),config,'cpu')
    assert evaluator.baseline['arcface'].app is evaluator.metrics['arcface'].app
    assert evaluator.baseline['musiq'].network is evaluator.metrics['musiq'].network
    for _ in range(2):
        report=evaluator.evaluate(torch.nn.Identity())
        assert report.summary['restoration']==report.summary['input']
        assert report.summary['restoration']['arcface']['coverage']==1
        assert len(report.per_image)==2
    report.save(tmp_path/'report')
    assert (tmp_path/'report/summary.json').exists()
