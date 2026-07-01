#!/bin/sh
# Deep K-SVD reconstruction training — one job per missing fraction.
# Submit via: sbatch train_deep_ksvd_reconstruction.sh 0.2
#             sbatch train_deep_ksvd_reconstruction.sh 0.4
#             sbatch train_deep_ksvd_reconstruction.sh 0.6

#SBATCH --job-name=dksvd-recon
#SBATCH --output=/d/hpc/home/sm79111/thesis/results/deep_ksvd/dksvd-recon-%j.out
#SBATCH --error=/d/hpc/home/sm79111/thesis/results/deep_ksvd/dksvd-recon-%j.err
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8

MF=${1:?Usage: sbatch train_deep_ksvd_reconstruction.sh <missing_fraction>}

cd /d/hpc/home/sm79111/thesis

LOG_DIR=/d/hpc/home/sm79111/thesis/results/deep_ksvd
mkdir -p "$LOG_DIR"
LOG_OUT="$LOG_DIR/dksvd-recon-mf${MF}-${SLURM_JOB_ID}.out"
LOG_ERR="$LOG_DIR/dksvd-recon-mf${MF}-${SLURM_JOB_ID}.err"

module load CUDA/12.6.0
source /d/hpc/home/sm79111/thesis/.venv_torch/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

{
echo "Node             : $SLURMD_NODENAME"
echo "Missing fraction : $MF"
echo "Start            : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun --nodes=1 --exclusive python -u \
    experiments/Deep-K-SVD/train_deep_ksvd_reconstruction.py \
    --missing_fraction "$MF" \
    --data_dir   experiments/Deep-K-SVD/gray \
    --train_list experiments/Deep-K-SVD/train_gray.txt \
    --test_list  experiments/Deep-K-SVD/test_gray.txt \
    --output_dir results/deep_ksvd/checkpoints_recon \
    --epochs 3 \
    --eval_every  10000 \
    --save_every 100000 \
    --num_workers 6 \
    --amp

echo "Finished : $(date)"
} > "$LOG_OUT" 2> "$LOG_ERR"

echo "Wrote logs to  : $LOG_OUT"
echo "Wrote errors to: $LOG_ERR"
