"""Distinguish SOC checkpoints from baseline training checkpoints."""
from bridge_restore.bridge import VEBridge
from bridge_restore.checkpoint import load_checkpoint
from bridge_restore.config import ModelConfig
from bridge_restore.models import build_model

from .sampling import SOCSampler


def load_soc_checkpoint(path):
    state = load_checkpoint(path)
    if state.get("kind") != "bridge_soc":
        raise ValueError("Expected a SOC checkpoint")
    return state


def load_student(path, device="cpu"):
    """Restore the trained student and its sampler without the teacher."""
    state = load_soc_checkpoint(path)
    bridge = VEBridge(**state["bridge"])
    student = build_model(ModelConfig(**state["architecture"]), bridge, state["image_size"])
    student.load_state_dict(state["student"])
    sampler = SOCSampler(bridge, **state["sampling"])
    return student.to(device).eval(), sampler
