"""Continuous approximation of JPEG quantization-bin consistency."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def block_dct(plane, basis):
    blocks = plane.unfold(-2, 8, 8).unfold(-2, 8, 8)
    return basis @ blocks @ basis.T


def interval_error(coefficients, indices, table):
    lower = table * (indices - .5)
    upper = table * (indices + .5)
    return (coefficients - coefficients.clamp(min=lower, max=upper)).square()


class JPEGConsistencyReward(nn.Module):
    def __init__(self, subsampling):
        super().__init__()
        self.chroma_stride = {0: (1, 1), 1: (1, 2), 2: (2, 2)}[subsampling]
        k = torch.arange(8, dtype=torch.float64)[:, None]
        n = torch.arange(8, dtype=torch.float64)
        basis = .5 * torch.cos(math.pi * k * (n + .5) / 8)
        basis[0] /= math.sqrt(2)
        self.register_buffer("basis", basis.float())
        self.register_buffer("color", torch.tensor([
            [.299, .587, .114],
            [-.168735892, -.331264108, .5],
            [.5, -.418687589, -.081312411],
        ]))

    def forward(self, image, batch):
        rgb = (image + 1) * 127.5
        planes = torch.einsum("ij,bjhw->bihw", self.color, rgb)
        error = image.new_zeros(len(image))
        for i, metadata in enumerate(batch["jpeg"]):
            plane = planes[:, i]
            if i == 0:
                plane = plane - 128
            else:
                sy, sx = self.chroma_stride
                if (sy, sx) != (1, 1):
                    plane = F.pad(plane, (0, -plane.shape[-1] % sx,
                                          0, -plane.shape[-2] % sy), mode="replicate")
                    plane = F.avg_pool2d(plane, (sy, sx))
            rows, cols = metadata["indices"].shape[1:3]
            pad_h, pad_w = rows * 8 - plane.shape[-2], cols * 8 - plane.shape[-1]
            if not (0 <= pad_h < 8 and 0 <= pad_w < 8):
                raise ValueError("JPEG block dimensions do not match the input and sampling factors")
            plane = F.pad(plane, (0, pad_w, 0, pad_h), mode="replicate")
            coefficients = block_dct(plane, self.basis)
            error = error + interval_error(
                coefficients, metadata["indices"], metadata["table"][:, None, None],
            ).flatten(1).sum(1)
        return -error
