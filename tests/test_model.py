from dataclasses import replace
import pytest
import torch
from bridge_restore.bridge import VEBridge
from bridge_restore.models import build_model


@pytest.mark.parametrize("mode", ["bm", "abm"])
def test_conditioning_and_frozen_input_gradient(tiny_config, mode):
    model = build_model(replace(tiny_config.model, mode=mode, use_checkpoint=True), VEBridge(), 8).eval()
    # ADM output projection starts at zero: make it nonzero to test the actual path.
    with torch.no_grad():
        model.network.out[-1].weight.normal_(std=.01)
    x = torch.randn(2, 3, 8, 8)
    t = torch.tensor([.2, .7])
    first = model(x, t, torch.zeros_like(x))
    second = model(x, t, torch.ones_like(x))
    if mode == "bm":
        assert torch.equal(first, second)
    else:
        assert not torch.allclose(first, second)
        with pytest.raises(ValueError, match="ABM"):
            model(x, t)
    model.requires_grad_(False)
    x.requires_grad_(True)
    gradient, = torch.autograd.grad(model(x, t, torch.zeros_like(x)).sum(), x)
    assert torch.isfinite(gradient).all()
