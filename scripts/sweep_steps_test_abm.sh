#!/usr/bin/env bash
#SBATCH --job-name=bridge-step-abm
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

cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
exec bash scripts/sweep_steps_test.sh \
    runs/celeba_main_b128_5ep/medium_abm_T1 \
    "runs/medium_abm_T1_step_sweep_${SLURM_JOB_ID}"
