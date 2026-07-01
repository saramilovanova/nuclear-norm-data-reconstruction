#!/bin/sh
# Submit all three Deep K-SVD reconstruction training jobs in parallel.
# Each trains at one missing fraction on its own GPU.
#
# Usage: ./submit_deep_ksvd_reconstruction.sh

mkdir -p /d/hpc/home/sm79111/thesis/results/deep_ksvd

JOB1=$(sbatch --parsable train_deep_ksvd_reconstruction.sh 0.2)
JOB2=$(sbatch --parsable train_deep_ksvd_reconstruction.sh 0.4)
JOB3=$(sbatch --parsable train_deep_ksvd_reconstruction.sh 0.6)

echo "Submitted:"
echo "  mf=0.2 → job $JOB1"
echo "  mf=0.4 → job $JOB2"
echo "  mf=0.6 → job $JOB3"
echo ""
echo "Monitor : squeue -u \$USER"
echo "Cancel  : scancel $JOB1 $JOB2 $JOB3"
