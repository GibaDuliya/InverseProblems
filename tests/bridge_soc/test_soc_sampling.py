import torch

from bridge_restore.bridge import VEBridge
from bridge_restore.sampling import SDESampler
from bridge_soc.adjoint_matching import _propagate_adjoint
from bridge_soc.sampling import SOCSampler


class Denoiser(torch.nn.Module):
    def forward(self, x, t, observation):
        assert (t > 0).all()
        return .4 * x + t[:, None, None, None] * observation


def test_legacy_and_soc_stochastic_steps():
    bridge = VEBridge(sigma_min=.07)
    baseline = SDESampler(bridge, num_steps=3, clip_output=False)
    soc = SOCSampler(bridge, num_steps=3, clip_output=False)
    xT = torch.randn(4, 3, 4, 4)
    model = Denoiser()
    grid = baseline.make_time_grid("cpu")
    reference = xT
    generator = torch.Generator().manual_seed(5)
    for t, next_t in zip(grid, grid[1:]):
        dt = next_t - t
        predicted = model(reference, t.expand(len(reference)), xT)
        drift = 2 * (reference - predicted) / t
        noise = torch.randn(reference.shape, generator=generator)
        reference = reference + drift * dt + (2*t).sqrt() * (-dt).sqrt() * noise

    denoised = model(reference, grid[-1].expand(len(reference)), xT)
    generator = torch.Generator().manual_seed(5)
    legacy = baseline.sample(model, xT, generator)
    torch.testing.assert_close(legacy.reconstruction, denoised, atol=0, rtol=0)

    generator = torch.Generator().manual_seed(5)
    rollout = soc.rollout(model, xT, torch.tensor([3, 0, 2, 0]), generator)
    torch.testing.assert_close(rollout.terminal, 2 * denoised - reference)
    generator = torch.Generator().manual_seed(5)
    sample = soc.sample(model, xT, generator, clip_output=True)
    torch.testing.assert_close(sample.reconstruction, rollout.terminal.clamp(-1, 1))
    assert sample.nfe == legacy.nfe == 4
    assert model.training


def test_active_adjoint_matches_single_images():
    bridge = VEBridge(sigma_min=.07)
    soc = SOCSampler(bridge, num_steps=3)
    model = Denoiser().eval()
    xT = torch.randn(5, 3, 4, 4)
    a_N = torch.randn_like(xT)
    for indices in (torch.tensor([3, 0, 2, 0, 3]), torch.full((5,), 3)):
        rollout = soc.rollout(model, xT, indices, torch.Generator().manual_seed(5))
        actual = _propagate_adjoint(model, bridge, xT, rollout, a_N)
        expected = []
        for k, adjoint in zip(indices.tolist(), a_N):
            for i in range(soc.num_steps - 1, k - 1, -1):
                t = rollout.grid[i + 1]
                dt = t - rollout.grid[i]
                adjoint = adjoint * (1 + dt * 2 * .6 / t)
            expected.append(adjoint)
        torch.testing.assert_close(actual, torch.stack(expected))
