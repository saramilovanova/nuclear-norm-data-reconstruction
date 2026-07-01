#!/bin/sh
# Submit all three Deep K-SVD training jobs in parallel.
# Each trains at one noise level on its own GPU.
#
# Usage: ./submit_deep_ksvd.sh

mkdir -p /d/hpc/home/sm79111/thesis/results/deep_ksvd

JOB1=$(sbatch --parsable train_deep_ksvd.sh 12.75)
JOB2=$(sbatch --parsable train_deep_ksvd.sh 25.5)
JOB3=$(sbatch --parsable train_deep_ksvd.sh 51.0)

echo "Submitted:"
echo "  sigma=12.75 → job $JOB1"
echo "  sigma=25.5  → job $JOB2"
echo "  sigma=51.0  → job $JOB3"
echo ""
echo "Monitor : squeue -u \$USER"
echo "Cancel  : scancel $JOB1 $JOB2 $JOB3"
