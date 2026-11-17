#!/usr/bin/env bash
#SBATCH --job-name=bridge-train
#SBATCH --partition=ais-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/gpfs0/ajnaz.gibadullin/InverseProblems/slurm-%x-%j.out
#SBATCH --error=/gpfs/gpfs0/ajnaz.gibadullin/InverseProblems/slurm-%x-%j.err

# Paths below allow submission from any working directory.
# Instructions: docs/slurm.md (local only).
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo 'Submit with sbatch scripts/train_slurm.sh; do not run with bash.' >&2
    exit 2
fi
mode=${1:-train}
case "$mode" in
    train) config=${2:-configs/train.yaml}; module=bridge_restore.train ;;
    grid) config=${2:-configs/celeba_main_grid.yaml}; module=bridge_restore.grid ;;
    *) echo 'Usage: sbatch scripts/train_slurm.sh [train|grid] [config.yaml]' >&2; exit 2 ;;
esac
if (( $# > 2 )); then
    echo 'Too many arguments.' >&2
    exit 2
fi
CONTAINER_IMAGE=/trinity/home/ajnaz.gibadullin/sif_containers/es-bench.sif
project_dir=/gpfs/gpfs0/ajnaz.gibadullin/InverseProblems
venv_dir=.venv
[[ -d "$project_dir/src/bridge_restore" ]] || { echo "Not a project root: $project_dir" >&2; exit 2; }
[[ -f "$CONTAINER_IMAGE" ]] || { echo "Container missing: $CONTAINER_IMAGE" >&2; exit 2; }
[[ -f "$project_dir/$config" && "$config" != /* ]] || { echo 'Config must be a project-relative existing file.' >&2; exit 2; }
if command -v apptainer >/dev/null 2>&1; then
    runtime=apptainer
elif command -v singularity >/dev/null 2>&1; then
    runtime=singularity
else
    echo 'Neither apptainer nor singularity available; load your cluster container module.' >&2
    exit 2
fi
# These must be exported before Python starts. No dependency installation or git
# update occurs on the compute node; prepare the existing environment in advance.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export PYTHONUNBUFFERED=1
printf 'Job=%s node=%s mode=%s config=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$mode" "$config"
printf 'Project=%s container=%s environment=%s\n' "$project_dir" "$CONTAINER_IMAGE" "$venv_dir"
# Starting through srun keeps the process within the allocated Slurm job step.
exec srun "$runtime" exec --nv --bind "$project_dir:/workspace" --pwd /workspace \
    "$CONTAINER_IMAGE" /bin/bash -c '
        set -euo pipefail
        source "$1/bin/activate"
        export OMP_NUM_THREADS="$4" MKL_NUM_THREADS="$5" PYTHONUNBUFFERED=1
        python -c "import torch; assert torch.cuda.is_available(), '\''CUDA unavailable'\''; print(torch.__version__, torch.cuda.get_device_name(0)); print(torch.ones(1, device='\''cuda'\''))"
        exec python -m "$2" --config "$3"
    ' bridge-job "$venv_dir" "$module" "$config" "$OMP_NUM_THREADS" "$MKL_NUM_THREADS"
