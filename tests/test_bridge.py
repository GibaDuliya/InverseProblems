import torch
from bridge_restore.bridge import VEBridge


def test_endpoint_marginals_and_midpoint_noise():
    bridge = VEBridge(sigma_max=2.)
    x0, xT = torch.zeros(20000, 1), torch.ones(20000, 1)
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(1))
    assert torch.equal(bridge.sample_marginal(x0, xT, torch.zeros(20000), noise), x0)
    assert torch.equal(bridge.sample_marginal(x0, xT, torch.full((20000,), 2.), noise), xT)
    t = torch.full((20000,), 2**.5)
    values = bridge.sample_marginal(x0, xT, t, noise)
    assert abs(values.mean().item() - .5) < .02
    assert abs(values.var().item() - 1.) < .03


def test_preconditioning_normalizes_target_variance():
    bridge = VEBridge(cov_xy=.12)
    t = torch.tensor([.01, .2, .6, .99], dtype=torch.float64)
    a = t.square()
    b = 1 - a
    noise_var = t.square() * b
    data_var = .25
    state_var = (a*a+b*b)*data_var + 2*a*b*.12 + noise_var
    cross = b*data_var+a*.12
    scales = bridge.preconditioning(t)
    residual_var = data_var + scales.c_skip.square()*state_var - 2*scales.c_skip*cross
    torch.testing.assert_close(scales.c_in.square()*state_var, torch.ones_like(t))
    torch.testing.assert_close(scales.c_out.square(), residual_var)


def test_reverse_drift_orientation():
    bridge = VEBridge()
    def zero(xt, t, observation=None):
        return torch.zeros_like(xt)
    x = torch.ones(2, 3, 4, 4)
    drift = bridge.reverse_drift(zero, x, torch.full((2,), .5), x)
    assert (x + drift * -.01 < x).all()
    torch.testing.assert_close(bridge.diffusion(torch.tensor([.5])), torch.ones(1))
