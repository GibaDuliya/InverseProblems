from dataclasses import asdict
import json

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from bridge_restore.bridge import VEBridge
from bridge_restore.checkpoint import load_checkpoint, save_checkpoint
from bridge_restore.data.build import build_data
from bridge_restore.models import build_model
from bridge_soc.checkpoint import load_student
from bridge_soc.train import main
from bridge_soc.training import SOCTrainer


@pytest.mark.parametrize("reward_input,clip_output", [("raw", True), ("clipped", False)])
def test_soc_cli_resume_and_inference(tiny_config, tmp_path, monkeypatch, reward_input, clip_output):
    pytest.importorskip("jpeglib")
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(57)
    for index in range(10):
        Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(images / f"{index}.png")
    tiny_config.model.mode = "bm"
    tiny_config.data.factory = "bridge_restore.data.datasets:ImageFolderDataset"
    tiny_config.data.kwargs = {"root": str(images)}
    tiny_config.split.mode = "create_if_missing"
    tiny_config.split.manifest_path = str(tmp_path / "split.json")
    tiny_config.split.group_by = "sample"
    tiny_config.split.fractions = [.5, .25, .25]
    tiny_config.sampling.num_steps = 2
    data = build_data(tiny_config.data, tiny_config.split, tiny_config.corruption, 2, 42)
    for split in ("val", "test"):
        ids = getattr(data.split_manifest, f"{split}_ids")
        (tmp_path / f"{split}.json").write_text(json.dumps({"split": split, "sample_ids": ids[:1]}))
    teacher = build_model(tiny_config.model, VEBridge(**asdict(tiny_config.bridge)), 8)
    teacher_path = tmp_path / "teacher.pt"
    save_checkpoint(teacher_path, {
        "format_version": 1, "config": asdict(tiny_config),
        "model": teacher.state_dict(), "ema": {"parameters": teacher.state_dict()},
        "split_checksum": data.split_manifest.checksum, "corruption_description": data.corruption.to_config(),
    })

    cfg = {"control": {"teacher_checkpoint": str(teacher_path), "mode": "abm"},
           "reward": {"name": "jpeg_consistency", "input": reward_input},
           "sampling": {"clip_output": clip_output},
           "evaluation": {"subset_size": 2, "test_subset_size": None, "batch_size": 2,
                          "visual_samples_path": str(tmp_path / "val.json"),
                          "test_visual_samples_path": str(tmp_path / "test.json")},
           "training": {"device": "cpu", "batch_size": 2, "grad_accum_steps": 2, "beta": 1000., "max_steps": 1,
                        "validation_interval": 1, "checkpoint_interval": 1}}
    config_path = tmp_path / "soc.yaml"

    def run(directory, steps, resume=None):
        cfg["output_dir"] = str(tmp_path / directory)
        cfg["training"]["max_steps"] = steps
        config_path.write_text(yaml.safe_dump(cfg))
        args = ["--config", str(config_path)]
        if resume:
            args.extend(["--resume", str(resume)])
        main(args)
        return tmp_path / directory / "checkpoints" / "last.pt"

    report = SOCTrainer._report

    def controlled_report(self, evaluator, split, step):
        result = report(self, evaluator, split, step)
        if split == "val":
            result.summary["restoration"]["reward"]["mean"] = {1: -1., 2: -1., 3: -2.}[step]
        return result

    monkeypatch.setattr(SOCTrainer, "_report", controlled_report)
    full = run("full", 3)
    publish = SOCTrainer._publish_best

    def interrupt_publish(self, state):
        raise RuntimeError("interrupted before best publication")

    monkeypatch.setattr(SOCTrainer, "_publish_best", interrupt_publish)
    with pytest.raises(RuntimeError, match="interrupted"):
        run("resume", 1)
    monkeypatch.setattr(SOCTrainer, "_publish_best", publish)
    first = tmp_path / "resume" / "checkpoints" / "last.pt"
    first_state = load_checkpoint(first)
    assert first_state["data_position"] == (0, 2)
    assert first_state["samples_seen"] == 4
    resumed = run("resume", 3, resume=first)
    resumed_state = load_checkpoint(resumed)
    full_state = load_checkpoint(full)
    resumed_weights = resumed_state["student"]
    for key in ("student", "optimizer", "rollout_rng", "data_position", "samples_seen", "best"):
        torch.testing.assert_close(resumed_state[key], full_state[key], rtol=0, atol=0)
    assert resumed_state["step"] == 3
    assert resumed_state["samples_seen"] == 9
    assert any(not torch.equal(first_state["student"][key], value) for key, value in resumed_weights.items())
    output = resumed.parents[1]
    best = output / "best" / "reward"
    assert load_checkpoint(best / "checkpoint.pt")["step"] == 1
    panel = "images/panel_00.png"
    assert (best / panel).read_bytes() == (output / "validation" / "step_00000001" / panel).read_bytes()
    test = output / "test" / "reward"
    protocol = json.loads((test / "protocol.json").read_text())
    assert protocol["step"] == 1
    assert set(protocol["sample_ids"]) == set(data.split_manifest.test_ids)
    rows = [json.loads(line) for line in (test / "per_image.jsonl").read_text().splitlines()]
    summary = json.loads((test / "summary.json").read_text())
    assert summary["restoration"]["reward"]["mean"] == sum(row["restoration/reward"] for row in rows) / len(rows)

    # Legacy states used one update per batch and contain no selection history.
    cfg["training"]["grad_accum_steps"] = 1
    legacy = run("legacy_source", 1)
    legacy_state = load_checkpoint(legacy)
    for key in ("soc_protocol_version", "best", "last_validation_step", "samples_seen", "evaluation_samples"):
        del legacy_state[key]
    legacy_state["config"]["training"].pop("grad_accum_steps")
    legacy_state["config"]["training"]["validation_batches"] = 4
    legacy_state["config"].pop("evaluation")
    save_checkpoint(legacy, legacy_state)
    cfg["training"]["validation_interval"] = 2
    migrated = run("legacy_migrated", 3, resume=legacy)
    migrated_state = load_checkpoint(migrated)
    assert migrated_state["best"] == {"step": 1, "value": -1.}
    assert migrated_state["last_validation_step"] == 3
    assert migrated_state["samples_seen"] == 5

    observation = next(iter(data.val_loader))["xT"]
    teacher_path.unlink()
    student, sampler = load_student(resumed)
    baseline, baseline_sampler = load_student(full)
    assert student.mode == "abm"
    torch.testing.assert_close(
        sampler.sample(student, observation, torch.Generator().manual_seed(22)).reconstruction,
        baseline_sampler.sample(baseline, observation, torch.Generator().manual_seed(22)).reconstruction,
        rtol=0, atol=0,
    )
