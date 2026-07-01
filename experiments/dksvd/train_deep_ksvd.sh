#!/bin/sh
# Deep K-SVD training — one job per sigma level.
# Submit via: sbatch train_deep_ksvd.sh 12.75
#             sbatch train_deep_ksvd.sh 25.5
#             sbatch train_deep_ksvd.sh 51.0
#
# Key differences from the classical k-SVD array job:
#   - No --array: parallelism is across sigma values (3 jobs), not images
#   - Longer wall time: deep training runs for many hours
#   - The three jobs are independent and can run simultaneously on separate GPUs

#SBATCH --job-name=dksvd-sigma
#SBATCH --output=/d/hpc/home/sm79111/thesis/results/deep_ksvd/dksvd-%j.out
#SBATCH --error=/d/hpc/home/sm79111/thesis/results/deep_ksvd/dksvd-%j.err
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8

SIGMA=${1:?Usage: sbatch train_deep_ksvd.sh <sigma>}

cd /d/hpc/home/sm79111/thesis

LOG_DIR=/d/hpc/home/sm79111/thesis/results/deep_ksvd
mkdir -p "$LOG_DIR"
LOG_OUT="$LOG_DIR/dksvd-sigma-${SIGMA}-${SLURM_JOB_ID}.out"
LOG_ERR="$LOG_DIR/dksvd-sigma-${SIGMA}-${SLURM_JOB_ID}.err"

# Activate the deep-learning venv (separate from the conda env used for classical methods)
module load CUDA/12.6.0   # what's available on cluster
source /d/hpc/home/sm79111/thesis/.venv_torch/bin/activate

export PYTHONUNBUFFERED=1
# GPU training: OMP threads matter less, but harmless to set
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

{
echo "Node     : $SLURMD_NODENAME"
echo "Sigma    : $SIGMA"
echo "Start    : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun --nodes=1 --exclusive python -u \
    experiments/Deep-K-SVD/train_deep_ksvd.py \
    --sigma "$SIGMA" \
    --data_dir experiments/Deep-K-SVD/gray \
    --train_list experiments/Deep-K-SVD/train_gray.txt \
    --test_list  experiments/Deep-K-SVD/test_gray.txt \
    --output_dir results/deep_ksvd/checkpoints \
    --epochs 3 \
    --eval_every 10000 \
    --save_every 100000 \
    --num_workers 6 \
    --amp

echo "Finished : $(date)"
} > "$LOG_OUT" 2> "$LOG_ERR"

echo "Wrote logs to: $LOG_OUT"
echo "Wrote errors to: $LOG_ERR"
