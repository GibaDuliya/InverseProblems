#!/usr/bin/env bash
#SBATCH --job-name=bridge-step-test
#SBATCH --partition=gpu_devel
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
run_dir=${1:-runs/celeba_main_b128_5ep/medium_bm_T1}
output=${2:-runs/medium_bm_T1_step_sweep_${SLURM_JOB_ID}}
image=${CONTAINER_IMAGE:-/trinity/home/ajnaz.gibadullin/sif_containers/es-bench.sif}
if (( $# > 2 )) || [[ ! -f "$project_dir/scripts/sweep_steps_test.py" || ! -f "$image" ]]; then
    echo 'Usage: submit from the repository root with the existing container available.' >&2
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
        exec python scripts/sweep_steps_test.py --run-dir "$1" --output "$2" --seed 42 --device cuda
    ' sweep-test "$run_dir" "$output"
