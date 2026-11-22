"""Evaluate five sampling grids on the 32 visual examples of the original LPIPS test."""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import monotonic

from PIL import Image, ImageDraw
import torch
from torch.utils.data import default_collate

from bridge_restore.bridge import VEBridge
from bridge_restore.checkpoint import load_checkpoint
from bridge_restore.config import config_from_dict
from bridge_restore.data import build_data
from bridge_restore.evaluation.report import EvaluationReport
from bridge_restore.metrics import MetricBatch, build_metrics
from bridge_restore.models import build_model
from bridge_restore.sampling import SDESampler


STEPS = (100, 250, 500, 750, 1000)
METRICS = ("lpips", "ssim", "psnr", "arcface", "observation_mse", "observation_mae", "musiq")
BATCH_SIZE = 32


def save_panels(output, clean, jpeg, restored):
    labels = [*(str(steps) for steps in STEPS), "JPEG", "clean"]
    columns = [*(restored[steps] for steps in STEPS), jpeg, clean]
    height, width = clean.shape[-2:]
    header = 24
    panel_dir = output / "panels"
    panel_dir.mkdir()
    for start in range(0, BATCH_SIZE, 4):
        panel = Image.new("RGB", (len(columns) * width, header + 4 * height), "white")
        draw = ImageDraw.Draw(panel)
        for col, (label, images) in enumerate(zip(labels, columns)):
            draw.text((col * width + width // 2, header // 2), label, fill="black", anchor="mm")
            for row, image in enumerate(images[start:start + 4]):
                pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
                panel.paste(Image.fromarray(pixels.permute(1, 2, 0).numpy()),
                            (col * width, header + row * height))
        panel.save(panel_dir / f"panel_{start // 4 + 1:02d}.png")


def sample_with_progress(model, sampler, jpeg, seed):
    steps = sampler.num_steps
    nfe = 0
    started = monotonic()
    print(f"SWEEP_START steps={steps}", flush=True)

    def report_progress(_model, _inputs, _outputs):
        nonlocal nfe
        nfe += 1
        if nfe % 25 == 0 or nfe == steps + 1:
            if jpeg.is_cuda:
                torch.cuda.synchronize(jpeg.device)
            print(f"SWEEP_PROGRESS steps={steps} nfe={nfe}/{steps + 1} seconds={monotonic() - started:.1f}",
                  flush=True)

    handle = model.register_forward_hook(report_progress)
    try:
        generator = torch.Generator(device=jpeg.device).manual_seed(seed)
        return sampler.sample(model, jpeg, generator=generator)
    finally:
        handle.remove()


@torch.no_grad()
def run(run_dir, output, seed, device):
    checkpoint = run_dir / "best/lpips/checkpoint.pt"
    test_protocol = run_dir / "test/lpips/protocol.json"
    reference = json.loads(test_protocol.read_text())
    ids = reference["visual_sample_ids"]
    if len(ids) != BATCH_SIZE or len(set(ids)) != BATCH_SIZE or not set(ids) <= set(reference["sample_ids"]):
        raise ValueError("Expected exactly 32 distinct visual IDs from the saved test")
    state = load_checkpoint(checkpoint)
    config = config_from_dict(state["config"])
    if config.model.mode not in ("bm", "abm") or config.bridge.sigma_max != 1.0:
        raise ValueError("Expected a BM or ABM checkpoint with T=1")
    if (reference["split"], reference["criterion"], reference["checkpoint_step"],
            reference["split_checksum"], reference["config"]["weights"]) != (
            "test", "lpips", state["step"], state["split_checksum"], "ema"):
        raise ValueError("Saved test protocol does not match the EMA LPIPS checkpoint")
    split = config.split if config.split.mode == "official" else replace(config.split, mode="load")
    data = build_data(config.data, split, config.corruption, BATCH_SIZE, config.seed)
    if (data.split_manifest.checksum != state["split_checksum"]
            or data.corruption.to_config() != state["corruption_description"]):
        raise ValueError("Dataset split or degradation differs from the checkpoint")
    dataset = data.test_loader.dataset
    lookup = {key: index for index, key in enumerate(dataset.sample_ids)}
    batch = default_collate([dataset[lookup[key]] for key in ids])
    if batch["sample_id"] != ids:
        raise ValueError("Loaded samples differ from the stored visual IDs")
    clean, jpeg = (batch[key].to(device, dtype=torch.float32) for key in ("x0", "xT"))
    bridge = VEBridge(**asdict(config.bridge))
    model = build_model(config.model, bridge, config.data.image_size).to(device, dtype=torch.float32)
    model.load_state_dict(state["ema"]["parameters"])
    model.eval()
    del state

    output.mkdir(parents=True, exist_ok=False)
    metric_config = {name: reference["config"]["metrics"][name] for name in METRICS}
    print(f"SWEEP_INIT samples={len(ids)} batch={BATCH_SIZE} seed={seed} output={output}", flush=True)
    metrics = build_metrics(metric_config, device, data.corruption)
    protocol = {
        "checkpoint": str(checkpoint), "checkpoint_step": reference["checkpoint_step"],
        "reference_protocol": str(test_protocol), "split_checksum": data.split_manifest.checksum,
        "sample_ids": ids, "seed": seed, "batch_size": BATCH_SIZE, "weights": "ema",
        "model_mode": config.model.mode, "sigma_max": config.bridge.sigma_max,
        "precision": "float32", "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(device) if device == "cuda" else device,
        "metrics": {name: metric.description() for name, metric in metrics.items()},
    }

    restored = {}
    with (output / "comparison.csv").open("w", newline="") as stream:
        table = csv.writer(stream)
        table.writerow(("steps", "nfe", "samples", *METRICS))
        for steps in STEPS:
            started = monotonic()
            options = {**reference["sampler"], "num_steps": steps}
            result = sample_with_progress(model, SDESampler(bridge, **options), jpeg, seed)
            restored[steps] = result.reconstruction.cpu()
            summary = {}
            rows = {key: {"sample_id": key} for key in ids}
            metric_batch = MetricBatch(result.reconstruction, clean, ids, jpeg)
            for name, metric in metrics.items():
                metric.reset()
                metric.update(metric_batch)
                scores = metric.compute()
                summary[name] = scores.summary
                for key, score in scores.per_image.items():
                    rows[key][name] = score
            EvaluationReport(
                {"samples": len(ids), "nfe": result.nfe, "restoration": summary},
                list(rows.values()), {**protocol, "sampler": options},
            ).save(output / f"steps_{steps}")
            table.writerow((steps, result.nfe, len(ids), *(summary[name]["mean"] for name in METRICS)))
            stream.flush()
            print(f"SWEEP_DONE steps={steps} seconds={monotonic() - started:.1f}", flush=True)
    save_panels(output, clean.cpu(), jpeg.cpu(), restored)
    print(f"SWEEP_COMPLETE panels=8 output={output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run(args.run_dir, args.output, args.seed, args.device)
