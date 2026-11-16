"""DDBM x0 parameterization; conditioning is the only BM/ABM switch."""
import torch
from torch import nn
from .adm import ADMUNet
from ..bridge import broadcast


class Denoiser(nn.Module):
    """In BM observation is ignored; in ABM it is mandatory.

    xt and observation are [B,3,H,W], normalized to [-1,1] at the data boundary.
    Internal xt may lie outside that range. t is [B], 0<t<=T.
    The raw observation is concatenated, while xt receives c_in (as in DDBM).
    """
    def __init__(self, network, bridge, mode):
        super().__init__()
        if mode not in ("bm", "abm"):
            raise ValueError("mode must be bm or abm")
        self.network = network
        self.bridge = bridge
        self.mode = mode

    def forward(self, xt, t, observation=None):
        if t.ndim != 1 or t.shape[0] != xt.shape[0]:
            raise ValueError("t must have shape [B]")
        if self.mode == "abm" and (observation is None or observation.shape != xt.shape):
            raise ValueError("ABM requires observation with same shape as xt")
        scales = self.bridge.preconditioning(t)
        result = self.network(broadcast(scales.c_in, xt) * xt, scales.c_noise,
                              xT=observation if self.mode == "abm" else None)
        return broadcast(scales.c_skip, xt) * xt + broadcast(scales.c_out, xt) * result


def build_model(config, bridge, image_size):
    network = ADMUNet(
        image_size=image_size, in_channels=3, out_channels=3,
        model_channels=config.channels, num_res_blocks=config.num_res_blocks,
        channel_mult=tuple(config.channel_mult),
        attention_resolutions=tuple(image_size // r for r in config.attention_resolutions),
        dropout=config.dropout, num_heads=config.num_heads,
        use_scale_shift_norm=True, resblock_updown=True,
        use_checkpoint=config.use_checkpoint, attention_type="legacy",
        condition_mode="concat" if config.mode == "abm" else None)
    return Denoiser(network, bridge, config.mode)
