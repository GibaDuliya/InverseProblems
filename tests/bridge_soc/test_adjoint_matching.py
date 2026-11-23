from types import SimpleNamespace

import torch

from bridge_restore.bridge import VEBridge
from bridge_soc.adjoint_matching import _loss_on_rollout, _terminal_adjoint


class Denoiser(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight))

    def forward(self, x, times, observation):
        return self.weight + times[:, None, None, None] * x


def test_terminal_adjoint_clipping_and_sign():
    terminal = torch.tensor([-2., .5, 2.]).reshape(3, 1, 1, 1)
    reward = lambda image, batch: image.flatten(1).sum(1)
    for mode, expected in [("raw", [-.5, -.5, -.5]), ("clipped", [0, -.5, 0])]:
        adjoint, _ = _terminal_adjoint(terminal, reward, {}, 2., mode)
        torch.testing.assert_close(adjoint.flatten(), torch.tensor(expected))


def test_quadrature_loss_and_gradient():
    bridge = VEBridge()
    teacher = Denoiser(.1).requires_grad_(False)
    student = Denoiser(.3)
    states = torch.tensor([.2, .4, .6, .8]).reshape(4, 1, 1, 1)
    grid = torch.tensor([.9, .5, .1, 0.])
    batch = {"xT": states[:1]}
    reward = lambda image, batch: image.flatten(1).sum(1)
    estimates = []
    gradients = []
    for k in range(3):
        rollout = SimpleNamespace(
            grid=grid, terminal=states[-1:], order=torch.tensor([0]),
            selected=states[k:k+1], active_counts=[int(k <= i) for i in range(2)],
            right_states=[states[i+1:i+2] for i in range(2)],
        )
        loss = _loss_on_rollout(teacher, student, bridge, reward, batch, rollout,
                                torch.tensor([k]), 1., "raw").loss
        estimates.append(loss.detach())
        gradients.append(torch.autograd.grad(loss, student.weight)[0])
    # Independent scalar recurrence uses the known analytic teacher Jacobian.
    adjoints = [-1.]
    for i in (1, 0):
        jacobian = 2 * (1 - grid[i+1]) / grid[i+1]
        adjoints.insert(0, adjoints[0] * (1 + (grid[i+1] - grid[i]) * jacobian))
    t = grid[:-1]
    full = ((t - grid[1:]) / t.pow(3) *
            (student.weight - .1 + t.square() * torch.tensor(adjoints)).square()).sum()
    torch.testing.assert_close(torch.stack(estimates).mean(), full)
    torch.testing.assert_close(torch.stack(gradients).mean(), torch.autograd.grad(full, student.weight)[0])
    assert teacher.weight.grad is None
