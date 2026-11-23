"""VE trajectories for SOC with a final deterministic Euler step."""
from dataclasses import dataclass

import torch

from bridge_restore.sampling import SamplingResult, positive_time_grid, stochastic_step


@dataclass
class SOCRollout:
    grid: torch.Tensor
    terminal: torch.Tensor
    selected: torch.Tensor
    order: torch.Tensor
    active_counts: list
    right_states: list


class SOCSampler:
    def __init__(self, bridge, num_steps=100, start_offset=.0001, clip_output=True):
        if num_steps < 1 or not 0 < start_offset < bridge.sigma_max - bridge.sigma_min:
            raise ValueError("Invalid sampling interval/steps")
        self.bridge = bridge
        self.num_steps = num_steps
        self.start_offset = start_offset
        self.clip_output = clip_output

    @property
    def num_intervals(self):
        return self.num_steps + 1

    def make_time_grid(self, device):
        positive = positive_time_grid(self.bridge, self.num_steps, self.start_offset, device)
        return torch.cat((positive, positive.new_zeros(1)))

    @torch.no_grad()
    def _generate(self, denoiser, observation, generator, indices=None):
        training = denoiser.training
        denoiser.eval()
        try:
            x = observation.clone()
            grid = self.make_time_grid(x.device)
            if indices is not None:
                order = torch.argsort(indices, stable=True)
                nodes = torch.arange(self.num_intervals, device=x.device)
                counts = torch.searchsorted(indices[order], nodes, right=True).tolist()
                boundaries = [0, *counts]
                selected = torch.empty_like(x)
                right_states = []

            for i, t in enumerate(grid[:-1]):
                if indices is not None:
                    chosen = order[boundaries[i]:boundaries[i + 1]]
                    selected[chosen] = x[chosen]
                    if i > 0:
                        right_states.append(x[order[:counts[i - 1]]])

                if i == self.num_steps:
                    drift = self.bridge.reverse_drift(denoiser, x, t.expand(len(x)), observation)
                    x = x + (grid[i + 1] - t) * drift
                else:
                    x = stochastic_step(self.bridge, denoiser, x, observation, t, grid[i + 1], generator)

            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite SOC reconstruction")
            if indices is not None:
                return SOCRollout(grid, x, selected, order, counts[:-1], right_states)
            return x
        finally:
            denoiser.train(training)

    def sample(self, denoiser, observation, generator=None, *, clip_output=None):
        terminal = self._generate(denoiser, observation, generator)
        clip = self.clip_output if clip_output is None else clip_output
        return SamplingResult(terminal.clamp(-1, 1) if clip else terminal, self.num_intervals)

    def rollout(self, denoiser, observation, indices, generator=None):
        return self._generate(denoiser, observation, generator, indices)
