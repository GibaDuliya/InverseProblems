"""Run the main BM/ABM training grid sequentially, one fresh process per model."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import yaml
from .config import config_from_dict, save_config


def configurations(spec):
    if set(spec) != {"times", "modes", "models", "experiment"}:
        raise ValueError("Grid requires times, modes, models, experiment")
    names = set()
    for model_name, model in spec["models"].items():
        for mode in spec["modes"]:
            for time in spec["times"]:
                values = copy.deepcopy(spec["experiment"])
                values["model"].update(model, mode=mode)
                values["bridge"]["sigma_max"] = time
                name = f"{model_name}_{mode}_T{time:g}"
                if Path(name).name != name or name in names:
                    raise ValueError("Invalid or duplicate grid name")
                names.add(name)
                values["output_dir"] = str(Path(values["output_dir"]) / name)
                yield name, config_from_dict(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--list", action="store_true", help="Validate and list without training")
    parser.add_argument("--only", nargs="+", help="Exact run names from --list")
    args = parser.parse_args()
    with open(args.config) as f:
        spec = yaml.safe_load(f)
    runs = list(configurations(spec))
    if args.only and set(args.only) - {name for name, _ in runs}:
        parser.error("Unknown --only run; inspect --list")
    runs = [(name, cfg) for name, cfg in runs if not args.only or name in args.only]
    for name, cfg in runs:
        print(f"{name}: {cfg.output_dir}", flush=True)
    if args.list:
        return
    # Fail before spending GPU time if any selected output already exists.
    for name, cfg in runs:
        output = Path(cfg.output_dir)
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Nonempty output {output}; use --only for remaining runs or bridge-train --resume")
    for name, cfg in runs:
        path = Path(spec["experiment"]["output_dir"]) / "resolved_configs" / f"{name}.yaml"
        save_config(cfg, path)
        subprocess.run([sys.executable, "-m", "bridge_restore.train", "--config", str(path)], check=True)


if __name__ == "__main__":
    main()
