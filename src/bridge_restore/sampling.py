"""Pure Euler–Maruyama in decreasing DDBM time, without hybrid/ODE guidance."""
from dataclasses import dataclass
import torch
from .bridge import broadcast


@dataclass
class SamplingResult:
    reconstruction: torch.Tensor
    nfe: int


def positive_time_grid(bridge, num_steps, start_offset, device):
    return torch.linspace(bridge.sigma_max - start_offset, bridge.sigma_min,
                          num_steps + 1, device=device)


def stochastic_step(bridge, denoiser, x, observation, t, next_t, generator):
    times = t.expand(x.shape[0])
    dt = next_t - t
    drift = bridge.reverse_drift(denoiser, x, times, observation)
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return x + drift * dt + broadcast(bridge.diffusion(times), x) * (-dt).sqrt() * noise


class SDESampler:
    """Approximate x_(T-offset) by observation; integrate to sigma_min.

    num_steps is the number of stochastic updates. One final denoiser call gives
    the clean estimate, so NFE=num_steps+1. Internal states are never clipped.
    start_offset is an absolute amount in author time, not a fraction of T.
    """
    def __init__(self, bridge, num_steps=100, start_offset=.0001, clip_output=True):
        if num_steps < 1 or not 0 < start_offset < bridge.sigma_max - bridge.sigma_min:
            raise ValueError("Invalid sampling interval/steps")
        self.bridge = bridge
        self.num_steps = num_steps
        self.start_offset = start_offset
        self.clip_output = clip_output

    def make_time_grid(self, device):
        return positive_time_grid(self.bridge, self.num_steps, self.start_offset, device)

    @torch.no_grad()
    def sample(self, denoiser, observation, generator=None):
        training = denoiser.training
        denoiser.eval()
        try:
            x = observation.clone()
            grid = self.make_time_grid(x.device)
            for t, next_t in zip(grid[:-1], grid[1:]):
                x = stochastic_step(self.bridge, denoiser, x, observation, t, next_t, generator)
            result = denoiser(x, grid[-1].expand(x.shape[0]), observation=observation)
            if not torch.isfinite(result).all():
                raise FloatingPointError("Nonfinite reconstruction; inspect scales/grid/checkpoint")
            if self.clip_output:
                result = result.clamp(-1, 1)
            return SamplingResult(result, self.num_steps + 1)
        finally:
            denoiser.train(training)
