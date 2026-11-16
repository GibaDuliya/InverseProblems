"""VE bridge in DDBM time: clean t=0, observation t=T; generation decreases t."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class Preconditioning:
    c_skip: torch.Tensor
    c_out: torch.Tensor
    c_in: torch.Tensor
    c_noise: torch.Tensor


def broadcast(values, target):
    return values.reshape(-1, *([1] * (target.ndim - 1)))


class VEBridge:
    """DDBM VE marginal and reverse SDE, with common endpoint scale sigma_data.

    sigma_max is T, not data standard deviation. cov_xy is a scalar endpoint
    covariance for preconditioning. The same positive loss weights serve BM/ABM.
    No model or trainable parameters are owned by this object.
    """
    def __init__(self, sigma_max=1., sigma_min=.001, sigma_data=.5, cov_xy=0.):
        if not all(math.isfinite(x) for x in (sigma_max, sigma_min, sigma_data, cov_xy)):
            raise ValueError("Nonfinite bridge parameters")
        if not 0 < sigma_min < sigma_max or sigma_data <= 0 or abs(cov_xy) >= sigma_data**2:
            raise ValueError("Invalid VE scales/covariance")
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.cov_xy = cov_xy

    def sample_time(self, batch_size, device, generator=None):
        # Uniform in author time t; independent of the sampling discretization.
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.rand(
            batch_size, device=device, generator=generator)

    def sample_marginal(self, x0, xT, t, noise=None):
        if x0.shape != xT.shape:
            raise ValueError("Endpoint shapes differ")
        if noise is None:
            noise = torch.randn_like(x0)
        t = broadcast(t, x0)
        a = (t / self.sigma_max).square()
        return (1 - a) * x0 + a * xT + t * (1 - a).clamp_min(0).sqrt() * noise

    def preconditioning(self, t):
        # Keep coefficients in float32/float64 even under mixed precision.
        a = (t / self.sigma_max).square()
        b = 1 - a
        variance = t.square() * b
        data_var = self.sigma_data**2
        total = (a.square() + b.square()) * data_var + 2 * a * b * self.cov_xy + variance
        c_in = total.rsqrt()
        c_skip = (b * data_var + a * self.cov_xy) / total
        c_out = ((a.square() * (data_var**2 - self.cov_xy**2) + data_var * variance) / total).sqrt()
        return Preconditioning(c_skip, c_out, c_in, 250 * t.log())

    def loss_weight(self, t):
        return self.preconditioning(t).c_out.square().reciprocal()

    def training_loss(self, denoiser, x0, xT, generator=None):
        """Return [B] weighted MSE. Callers control batch averaging/accumulation."""
        t = self.sample_time(x0.shape[0], x0.device, generator)
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        xt = self.sample_marginal(x0, xT, t, noise)
        prediction = denoiser(xt, t, observation=xT)
        mse = (prediction.float() - x0.float()).square().flatten(1).mean(1)
        return mse * self.loss_weight(t)

    def reverse_drift(self, denoiser, xt, t, observation):
        """Coefficient multiplying NEGATIVE dt. Do not negate it again in sampler."""
        prediction = denoiser(xt, t, observation=observation)
        return 2 * (xt - prediction) / broadcast(t, xt)

    def diffusion(self, t):
        return (2 * t).sqrt()
