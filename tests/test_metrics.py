import torch
from bridge_restore.metrics import MetricBatch, build_metrics

def test_pixel_metrics_weight_images_and_reset():
    metrics = build_metrics({'mse': {}, 'psnr': {}}, 'cpu', None)
    for i, delta in enumerate([.2, .4, .6]):
        batch = MetricBatch(torch.full((1,3,8,8), delta), torch.zeros(1,3,8,8), [str(i)])
        for metric in metrics.values(): metric.update(batch)
    assert abs(metrics['mse'].compute().summary['mean'] - (.01+.04+.09)/3) < 1e-7
    expected = -10*torch.tensor([.01,.04,.09]).log10().mean()
    assert abs(metrics['psnr'].compute().summary['mean'] - expected) < 1e-5
    metrics['mse'].reset()
    assert not metrics['mse'].values
