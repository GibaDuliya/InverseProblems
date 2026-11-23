"""One Adjoint Matching regression on trajectories of the current student."""
from dataclasses import dataclass
import math

import torch

from bridge_restore.bridge import broadcast


@dataclass
class AMResult:
    loss: torch.Tensor
    reward_values: torch.Tensor


def reward_image(terminal, mode):
    if mode == "raw":
        return terminal
    if mode == "clipped":
        return terminal.clamp(-1, 1)
    raise ValueError("reward input must be raw or clipped")


def _terminal_adjoint(terminal, reward, batch, beta, reward_input):
    with torch.enable_grad():
        terminal = terminal.detach().requires_grad_(True)
        values = reward(reward_image(terminal, reward_input), batch)
        if values.shape != (terminal.shape[0],):
            raise ValueError("Reward must return one scalar per image")
        adjoint, = torch.autograd.grad(-values.sum() / beta, terminal)
    return adjoint.detach(), values.detach()


def _propagate_adjoint(teacher, bridge, observation, rollout, terminal_adjoint):
    order = rollout.order
    observation = observation[order]
    # Identity transport on the last interval ending at zero.
    adjoint = terminal_adjoint[order]
    targets = torch.empty_like(adjoint)
    for i in range(len(rollout.active_counts) - 1, -1, -1):
        count = rollout.active_counts[i]
        targets[order[count:len(adjoint)]] = adjoint[count:]
        adjoint = adjoint[:count]
        if count == 0:
            break

        with torch.enable_grad():
            x = rollout.right_states[i].detach().requires_grad_(True)
            right_time = rollout.grid[i + 1].expand(count)
            drift = bridge.reverse_drift(teacher, x, right_time, observation[:count])
            vjp, = torch.autograd.grad(drift, x, grad_outputs=adjoint)
        dt = rollout.grid[i + 1] - rollout.grid[i]
        adjoint = (adjoint + dt * vjp).detach()
    targets[order[:len(adjoint)]] = adjoint
    return targets


def adjoint_matching_loss(
    teacher, student, sampler, reward, batch, *, beta, reward_input, generator=None,
):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    observation = batch["xT"]
    indices = torch.randint(
        sampler.num_intervals, (len(observation),),
        device=observation.device, generator=generator,
    )
    with torch.no_grad():
        rollout = sampler.rollout(student, observation, indices, generator=generator)
    return _loss_on_rollout(teacher, student, sampler.bridge, reward, batch, rollout,
                            indices, beta, reward_input)


def _loss_on_rollout(teacher, student, bridge, reward, batch, rollout,
                     indices, beta, reward_input):
    observation = batch["xT"]
    adjoint, values = _terminal_adjoint(rollout.terminal, reward, batch, beta, reward_input)
    targets = _propagate_adjoint(teacher, bridge, observation, rollout, adjoint)
    times = rollout.grid[indices]
    with torch.no_grad():
        base = teacher(rollout.selected, times, observation=observation)
    prediction = student(rollout.selected, times, observation=observation)
    residual = prediction.float() - base.float() + broadcast(times.square(), prediction) * targets.float()
    weights = (len(rollout.grid) - 1) * (times - rollout.grid[indices + 1]) / times.pow(3)
    return AMResult((weights * residual.square().flatten(1).sum(1)).mean(), values)
