from dataclasses import asdict

import pytest
import torch

from bridge_restore.bridge import VEBridge
from bridge_restore.checkpoint import save_checkpoint
from bridge_restore.models import build_model
from bridge_soc import ControlConfig, initialize_control_denoiser, load_control_config, load_teacher


@pytest.mark.parametrize("teacher_weights", ["ema", "raw"])
def test_load_teacher_from_checkpoint(tiny_config, tmp_path, teacher_weights):
    source = build_model(tiny_config.model, VEBridge(**asdict(tiny_config.bridge)), 8)
    ema = {name: value.clone() for name, value in source.state_dict().items()}
    key = "network.out.2.bias"
    ema[key].fill_(1)
    checkpoint = tmp_path / "teacher.pt"
    save_checkpoint(checkpoint, {
        "format_version": 1,
        "config": asdict(tiny_config),
        "ema": {"parameters": ema},
        "model": source.state_dict(),
    })
    path = tmp_path / "soc.yaml"
    yaml = f"teacher_checkpoint: '{checkpoint}'\n"
    if teacher_weights == "raw":
        yaml += "teacher_weights: raw\n"
    path.write_text(yaml, encoding="utf-8")

    config = load_control_config(path)
    teacher = load_teacher(config)
    student = initialize_control_denoiser(teacher, config)

    expected_weights = ema if teacher_weights == "ema" else source.state_dict()
    torch.testing.assert_close(teacher.state_dict(), expected_weights)
    torch.testing.assert_close(student.state_dict(), expected_weights)
    assert teacher.mode == student.mode == tiny_config.model.mode
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


@pytest.mark.parametrize(
    ("teacher_mode", "student_mode"),
    [("bm", "bm"), ("abm", "abm"), ("bm", "abm")],
)
def test_student_preserves_teacher_output(tiny_config, teacher_mode, student_mode):
    tiny_config.model.mode = teacher_mode
    teacher = build_model(tiny_config.model, VEBridge(), 8).eval().requires_grad_(False)
    with torch.no_grad():
        teacher.network.out[-1].weight.normal_(std=.01)

    student = initialize_control_denoiser(teacher, ControlConfig("unused.pt", mode=student_mode))
    assert student.mode == student_mode
    assert not student.training
    assert all(parameter.requires_grad for parameter in student.parameters())

    xt = torch.randn(2, 3, 8, 8)
    t = torch.tensor([.2, .7])
    with torch.no_grad():
        for observation in (torch.randn_like(xt), torch.randn_like(xt)):
            torch.testing.assert_close(student(xt, t, observation), teacher(xt, t, observation))

    observation = torch.randn_like(xt)
    original = {name: value.clone() for name, value in teacher.state_dict().items()}
    optimizer = torch.optim.SGD(student.parameters(), lr=.01)
    student(xt, t, observation).square().mean().backward()
    if teacher_mode == "bm" and student_mode == "abm":
        observation_weights = student.network.input_blocks[0][0].weight
        assert torch.count_nonzero(observation_weights[:, 3:]) == 0
        assert torch.count_nonzero(observation_weights.grad[:, 3:]) > 0

    optimizer.step()
    torch.testing.assert_close(teacher.state_dict(), original, rtol=0, atol=0)


def test_reject_observation_conditioned_teacher_for_bm_student(tiny_config):
    teacher = build_model(tiny_config.model, VEBridge(), 8)
    with pytest.raises(ValueError, match="ABM -> BM"):
        initialize_control_denoiser(teacher, ControlConfig("unused.pt", mode="bm"))
