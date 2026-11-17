import torch
from bridge_restore.bridge import VEBridge
from bridge_restore.sampling import SDESampler


class RecordingDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []
    def forward(self, x, t, observation=None):
        self.inputs.append(x.clone())
        return torch.zeros_like(x)


def test_euler_maruyama_noise_scale_and_nfe():
    bridge = VEBridge(sigma_min=.2)
    sampler = SDESampler(bridge, num_steps=1, start_offset=.1, clip_output=False)
    denoiser = RecordingDenoiser()
    x = torch.zeros(20000, 1, 1, 1)
    result = sampler.sample(denoiser, x, torch.Generator().manual_seed(8))
    # t=.9 -> .2: Var(g(t)*sqrt(abs(dt))*noise)=1.8*.7.
    assert abs(denoiser.inputs[-1].var().item() - 1.26) < .04
    assert result.nfe == len(denoiser.inputs) == 2
    assert denoiser.training  # restore original module mode
    assert torch.equal(result.reconstruction, torch.zeros_like(x))
