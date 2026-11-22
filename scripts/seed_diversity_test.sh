#!/usr/bin/env bash
#SBATCH --job-name=bridge-seed-div
#SBATCH --partition=gpu_devel
#SBATCH --nodelist=vt01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=geforce:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}
source_root=${1:-runs/celeba_main_b128_5ep}
k=${3:-16}
model=${4:-}
output=${2:-runs/seed_diversity_${model:-medium}_${SLURM_JOB_ID}}
image=${CONTAINER_IMAGE:-/trinity/home/ajnaz.gibadullin/sif_containers/es-bench.sif}

if (( $# > 4 )) || [[ ! -f "$project_dir/scripts/seed_diversity_test.py" || ! -f "$image" ]]; then
    echo 'Usage: sbatch scripts/seed_diversity_test.sh [source_root] [output] [k] [model]' >&2
    exit 2
fi
if command -v apptainer >/dev/null 2>&1; then
    runtime=apptainer
elif command -v singularity >/dev/null 2>&1; then
    runtime=singularity
else
    echo 'Neither apptainer nor singularity is available.' >&2
    exit 2
fi

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
exec srun "$runtime" exec --nv --bind "$project_dir:/workspace" --pwd /workspace "$image" \
    /bin/bash -c '
        set -euo pipefail
        source .venv/bin/activate
        export PYTHONPATH="/workspace/src:${PYTHONPATH:-}"
        python -c "import torch; name = torch.cuda.get_device_name(0); print(name); assert \"1080ti\" in name.lower().replace(\" \", \"\"), name; print(torch.ones(1, device=\"cuda\"))"
        args=(--source-root "$1" --output "$2" --k "$3")
        if [[ -n "$4" ]]; then args+=(--model "$4"); fi
        exec python scripts/seed_diversity_test.py "${args[@]}"
    ' seed-diversity "$source_root" "$output" "$k" "$model"
