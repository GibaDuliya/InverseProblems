#!/usr/bin/env bash
#SBATCH --job-name=bridge-abm-15ep
#SBATCH --partition=ais-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/gpfs/gpfs0/ajnaz.gibadullin/InverseProblems/slurm-%x-%A_%a.out
#SBATCH --error=/gpfs/gpfs0/ajnaz.gibadullin/InverseProblems/slurm-%x-%A_%a.err

set -euo pipefail

case "${SLURM_ARRAY_TASK_ID:-}" in
    0) config=configs/celeba_medium_abm_T1.yaml ;;
    1) config=configs/celeba_medium_abm_T1.25.yaml ;;
    2) config=configs/celeba_medium_abm_T1.5.yaml ;;
    *) echo 'Submit with sbatch scripts/train_abm_slurm.sh' >&2; exit 2 ;;
esac

# Slurm copies this script to its spool. Resolve the training launcher from the checkout.
project_dir=/gpfs/gpfs0/ajnaz.gibadullin/InverseProblems
exec bash "$project_dir/scripts/train_slurm.sh" train "$config"
