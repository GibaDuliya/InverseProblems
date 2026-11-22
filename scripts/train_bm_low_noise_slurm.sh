#!/usr/bin/env bash
#SBATCH --job-name=bridge-bm-low-noise
#SBATCH --partition=ais-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/gpfs/gpfs0/ajnaz.gibadullin/InverseProblems/slurm-%x-%A_%a.out
#SBATCH --error=/gpfs/gpfs0/ajnaz.gibadullin/InverseProblems/slurm-%x-%A_%a.err

set -euo pipefail

case "${SLURM_ARRAY_TASK_ID:-}" in
    0) config=configs/celeba_medium_bm_T0.6.yaml ;;
    1) config=configs/celeba_medium_bm_T0.8.yaml ;;
    *) echo 'Submit with sbatch scripts/train_bm_low_noise_slurm.sh' >&2; exit 2 ;;
esac

# Slurm copies this script to its spool. Resolve the training launcher from the checkout.
project_dir=/gpfs/gpfs0/ajnaz.gibadullin/InverseProblems
exec bash "$project_dir/scripts/train_slurm.sh" train "$config"
