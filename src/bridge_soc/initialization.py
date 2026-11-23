"""Load the reference denoiser and initialize its trainable copy."""
from copy import deepcopy
from dataclasses import asdict
from typing import Union

import torch
from torch import nn

from bridge_restore.bridge import VEBridge
from bridge_restore.checkpoint import load_checkpoint
from bridge_restore.config import config_from_dict
from bridge_restore.models import Denoiser, build_model

from .config import ControlConfig


def load_teacher(
    config: ControlConfig,
    device: Union[str, torch.device] = "cpu",
) -> Denoiser:
    """Load and freeze the reference denoiser from a trusted checkpoint."""
    state = load_checkpoint(config.teacher_checkpoint)
    return teacher_from_state(state, config.teacher_weights, device)


def teacher_from_state(state, weights_kind, device="cpu"):
    teacher_config = config_from_dict(state["config"])
    bridge = VEBridge(**asdict(teacher_config.bridge))
    teacher = build_model(
        teacher_config.model,
        bridge,
        teacher_config.data.image_size,
    )

    if weights_kind == "ema":
        weights = state["ema"]["parameters"]
    else:
        weights = state["model"]

    teacher.load_state_dict(weights)
    return teacher.to(device).requires_grad_(False).eval()


def initialize_control_denoiser(
    teacher: Denoiser,
    config: ControlConfig,
) -> Denoiser:
    """Return an independent trainable denoiser with the teacher's initial output."""
    mode = teacher.mode if config.mode is None else config.mode
    if teacher.mode == "abm" and mode == "bm":
        raise ValueError("ABM -> BM cannot preserve the teacher's observation dependence")

    student = deepcopy(teacher)
    if teacher.mode == "bm" and mode == "abm":
        input_conv = student.network.input_blocks[0][0]
        state_channels = input_conv.in_channels
        conditioned_conv = nn.Conv2d(
            in_channels=2 * state_channels,
            out_channels=input_conv.out_channels,
            kernel_size=input_conv.kernel_size,
            stride=input_conv.stride,
            padding=input_conv.padding,
            dilation=input_conv.dilation,
            bias=input_conv.bias is not None,
            padding_mode=input_conv.padding_mode,
            device=input_conv.weight.device,
            dtype=input_conv.weight.dtype,
        )

        with torch.no_grad():
            conditioned_conv.weight[:, :state_channels].copy_(input_conv.weight)
            conditioned_conv.weight[:, state_channels:].zero_()
            if input_conv.bias is not None:
                conditioned_conv.bias.copy_(input_conv.bias)

        student.network.input_blocks[0][0] = conditioned_conv
        student.network.condition_mode = "concat"
        student.mode = "abm"

    return student.requires_grad_(True).eval()
