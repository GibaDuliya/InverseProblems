"""Measure pairwise diversity among seeded restorations of the original 32 visual test inputs."""
import argparse
import csv
from dataclasses import asdict, replace
import fcntl
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from tempfile import TemporaryDirectory
from time import monotonic

from PIL import Image
import torch
from torch.utils.data import default_collate

from bridge_restore.bridge import VEBridge
from bridge_restore.checkpoint import load_checkpoint, save_checkpoint
from bridge_restore.config import config_from_dict
from bridge_restore.data import build_data
from bridge_restore.metrics import MetricBatch, build_metrics
from bridge_restore.models import build_model
from bridge_restore.sampling import SDESampler


DEFAULT_MODELS = {
    f"medium_{mode}_T{limit:g}": (mode, limit, int(100 * limit))
    for mode in ("bm", "abm") for limit in (1, 1.5, 2)
}
MODELS = {**DEFAULT_MODELS, "medium_bm_T0.6": ("bm", 0.6, 60)}
METRICS = ("lpips", "ssim", "psnr")
SEEDS = tuple(range(42, 66))
BATCH_SIZE = 32


def read_sources(source_root, model=None):
    references = {}
    sources = {}
    selected = DEFAULT_MODELS if model is None else {model: MODELS[model]}
    for name, (mode, limit, steps) in selected.items():
        root = source_root / name
        reference_file = root / "test/lpips/protocol.json"
        reference = json.loads(reference_file.read_text())
        checkpoint = root / "best/lpips/checkpoint.pt"
        info = checkpoint.stat()
        if (reference["split"], reference["criterion"], reference["config"]["weights"]) != ("test", "lpips", "ema"):
            raise ValueError(f"Expected an EMA LPIPS test report: {reference_file}")
        ids = reference["visual_sample_ids"]
        if (len(ids) != BATCH_SIZE or len(set(ids)) != BATCH_SIZE
                or not set(ids) <= set(reference["sample_ids"])
                or any(Path(key).name != key or Path(key).stem in ("", ".", "..") for key in ids)
                or len({Path(key).stem for key in ids}) != BATCH_SIZE):
            raise ValueError(f"Expected 32 distinct visual IDs in {reference_file}")
        references[name] = reference
        sources[name] = {
            "mode": mode, "T": limit, "checkpoint": str(checkpoint.resolve()),
            "checkpoint_step": reference["checkpoint_step"], "checkpoint_bytes": info.st_size,
            "checkpoint_mtime_ns": info.st_mtime_ns,
            "test_protocol": str(reference_file.resolve()),
            "sampler": {**reference["sampler"], "num_steps": steps},
        }

    first = next(iter(references.values()))
    metrics = {metric: first["config"]["metrics"][metric] for metric in METRICS}
    for name, reference in references.items():
        if (reference["visual_sample_ids"] != first["visual_sample_ids"]
                or reference["split_checksum"] != first["split_checksum"]
                or any(reference["config"]["metrics"][metric] != metrics[metric] for metric in METRICS)):
            raise ValueError(f"Visual IDs, split or metric settings differ for {name}")
    return sources, first["visual_sample_ids"], first["split_checksum"], metrics


def save_png(path, image):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = ((image.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    temporary = path.with_suffix(".tmp")
    Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(temporary, format="PNG")
    temporary.replace(path)


def restorations_for_model(name, source, config, state, jpeg, output, ids, seeds, device):
    cache = output / ".fp32" / name
    cache.mkdir(parents=True, exist_ok=True)
    paths = [cache / f"seed_{seed:06d}.pt" for seed in seeds]
    if any(not path.is_file() for path in paths):
        bridge = VEBridge(**asdict(config.bridge))
        model = build_model(config.model, bridge, config.data.image_size).to(device, dtype=torch.float32)
        model.load_state_dict(state["ema"]["parameters"])
        model.eval()
        sampler = SDESampler(bridge, **source["sampler"])
    results = []
    for index, (seed, path) in enumerate(zip(seeds, paths), start=1):
        started = monotonic()
        if path.is_file():
            images = torch.load(path, map_location="cpu", weights_only=True)["restoration"]
        else:
            print(f"DIVERSITY_SEED_START model={name} seed={seed} index={index}/{len(seeds)}", flush=True)
            generator = torch.Generator(device=device).manual_seed(seed)
            images = sampler.sample(model, jpeg, generator=generator).reconstruction.cpu()
            save_checkpoint(path, {"restoration": images})
        if images.shape != jpeg.shape or images.dtype != torch.float32 or not torch.isfinite(images).all():
            raise ValueError(f"Invalid restoration batch for {name}, seed {seed}")
        for key, image in zip(ids, images):
            save_png(output / "restorations" / name / Path(key).stem / f"seed_{seed:06d}.png", image)
        results.append(images)
        print(f"DIVERSITY_SEED_DONE model={name} seed={seed} index={index}/{len(seeds)} "
              f"seconds={monotonic() - started:.1f}", flush=True)
    return results


def write_metrics(name, results, ids, seeds, metrics, device, pair_writer, image_writer, model_writer):
    per_image = {key: {metric: [] for metric in METRICS} for key in ids}
    for left, right in combinations(range(len(seeds)), 2):
        batch = MetricBatch(results[left].to(device), results[right].to(device), ids)
        scores = {
            key: metric.values_for(batch).detach().cpu().reshape(-1).tolist()
            for key, metric in metrics.items()
        }
        if any(len(values) != len(ids) for values in scores.values()):
            raise ValueError("Expected one metric value per image and seed pair")
        for index, key in enumerate(ids):
            values = [scores[metric][index] for metric in METRICS]
            pair_writer.writerow((name, key, seeds[left], seeds[right], *values))
            for metric, value in zip(METRICS, values):
                if not math.isfinite(value) and not (metric == "psnr" and value == math.inf):
                    raise FloatingPointError(f"Invalid {metric} for {name} {key}: {value}")
                per_image[key][metric].append(value)

    averages = {metric: [] for metric in METRICS}
    for key, metrics_by_image in per_image.items():
        summary = []
        for metric, values in metrics_by_image.items():
            mean = fmean(values)
            averages[metric].append(mean)
            deviation = pstdev(values) if math.isfinite(mean) else math.nan
            summary.extend((mean, deviation))
        image_writer.writerow((name, key, *summary))
    model_writer.writerow((name, *(fmean(averages[metric]) for metric in METRICS)))


@torch.no_grad()
def run(source_root, output, seeds, device, model=None):
    sources, ids, split_checksum, metric_config = read_sources(source_root, model)
    metrics = build_metrics(metric_config, device, None)
    provenance = {
        "models": sources, "sample_ids": ids, "split_checksum": split_checksum,
        "batch_size": BATCH_SIZE, "seed_plan": SEEDS, "precision": "float32",
        "metrics": {name: metric.description() for name, metric in metrics.items()},
        "pair_standard_deviation": "population", "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(device) if device == "cuda" else device,
    }
    cache_manifest = output / ".fp32/protocol.json"
    serialized = json.dumps(provenance, indent=2, sort_keys=True)
    if cache_manifest.is_file() and cache_manifest.read_text() != serialized:
        raise ValueError(f"Inputs or runtime changed. Use a new output directory: {output}")
    complete = output / "protocol.json"
    if complete.is_file() and json.loads(complete.read_text())["k"] > len(seeds):
        raise ValueError("Refusing to replace a completed report with fewer seeds")
    temporary_manifest = cache_manifest.with_suffix(".tmp")
    temporary_manifest.write_text(serialized)
    temporary_manifest.replace(cache_manifest)

    print(f"DIVERSITY_INIT models={len(sources)} samples={len(ids)} k={len(seeds)} output={output}", flush=True)
    with TemporaryDirectory(dir=output, prefix=".reports-") as temp:
        reports = Path(temp)
        with ((reports / "per_pair.csv").open("w", newline="") as pairs,
              (reports / "per_image.csv").open("w", newline="") as images,
              (reports / "per_model.csv").open("w", newline="") as models):
            pair_writer, image_writer, model_writer = map(csv.writer, (pairs, images, models))
            pair_writer.writerow(("model", "sample_id", "seed_1", "seed_2", *METRICS))
            image_writer.writerow(("model", "sample_id", *(
                f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")
            )))
            model_writer.writerow(("model", *METRICS))
            inputs_protocol = None
            for name, source in sources.items():
                started = monotonic()
                state = load_checkpoint(source["checkpoint"])
                config = config_from_dict(state["config"])
                if (config.model.mode, config.bridge.sigma_max, state["step"], state["split_checksum"]) != (
                        source["mode"], source["T"], source["checkpoint_step"], split_checksum):
                    raise ValueError(f"Checkpoint config or split differs from the reference for {name}")
                current = (config.data, config.split, config.corruption, state["corruption_description"])
                if inputs_protocol is None:
                    split = config.split if config.split.mode == "official" else replace(config.split, mode="load")
                    data = build_data(config.data, split, config.corruption, BATCH_SIZE, config.seed)
                    if (data.split_manifest.checksum != split_checksum
                            or data.corruption.to_config() != state["corruption_description"]):
                        raise ValueError("Dataset split or degradation differs from the checkpoint")
                    dataset = data.test_loader.dataset
                    lookup = {key: index for index, key in enumerate(dataset.sample_ids)}
                    batch = default_collate([dataset[lookup[key]] for key in ids])
                    if batch["sample_id"] != ids:
                        raise ValueError("Dataset IDs or order differ from the visual protocol")
                    for key, clean, observation in zip(ids, batch["x0"], batch["xT"]):
                        folder = output / "references" / Path(key).stem
                        save_png(folder / "clean.png", clean)
                        save_png(folder / "JPEG.png", observation)
                    jpeg = batch["xT"].to(device, dtype=torch.float32)
                    inputs_protocol = current
                elif inputs_protocol != current:
                    raise ValueError(f"Input preprocessing or degradation differs for {name}")

                results = restorations_for_model(name, source, config, state, jpeg, output, ids, seeds, device)
                del state
                print(f"DIVERSITY_METRICS model={name} pairs_per_image={math.comb(len(seeds), 2)}", flush=True)
                write_metrics(name, results, ids, seeds, metrics, device, pair_writer, image_writer, model_writer)
                del results
                print(f"DIVERSITY_MODEL_DONE model={name} seconds={monotonic() - started:.1f}", flush=True)

        (reports / "protocol.json").write_text(json.dumps(
            {**provenance, "seeds": seeds, "k": len(seeds)}, indent=2, sort_keys=True,
        ))
        for name in ("per_pair.csv", "per_image.csv", "per_model.csv", "protocol.json"):
            (reports / name).replace(output / name)
    print(f"DIVERSITY_COMPLETE models={len(sources)} samples={len(ids)} k={len(seeds)} output={output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--model", choices=MODELS, help="Run only the named model")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not 2 <= args.k <= len(SEEDS):
        parser.error(f"--k must be between 2 and {len(SEEDS)}")
    output = args.output.resolve()
    source_root = args.source_root.resolve()
    if output == source_root or source_root in output.parents:
        parser.error("--output must be outside the source experiment directory")
    cache = output / ".fp32"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(source_root, output, SEEDS[:args.k], args.device, args.model)


if __name__ == "__main__":
    main()
