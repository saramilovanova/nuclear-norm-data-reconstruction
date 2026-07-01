#!/bin/sh
# Deep K-SVD recommender reconstruction.
# Submit via: sbatch run_deep_ksvd_recommender_reconstruction.sh 0.4

#SBATCH --job-name=dksvd-recom-recon-${1}
#SBATCH --output=/d/hpc/home/sm79111/thesis/results/deep_ksvd/dksvd-recom-recon-mf${1}-%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=16G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

MF=${1:?Usage: sbatch run_deep_ksvd_recommender_reconstruction.sh <missing_fraction>}

cd /d/hpc/home/sm79111/thesis

module load CUDA/12.6.0
source /d/hpc/home/sm79111/thesis/.venv_torch/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Node             : $SLURMD_NODENAME"
echo "Missing fraction : $MF"
echo "Start            : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun --nodes=1 --exclusive python -u \
    experiments/Deep-K-SVD/deep_ksvd_recommender_reconstruction_multiseed.py \
    --missing_fraction "$MF"

echo "Finished : $(date)"
