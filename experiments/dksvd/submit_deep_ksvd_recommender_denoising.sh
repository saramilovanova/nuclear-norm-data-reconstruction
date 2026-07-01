#!/bin/sh
# Submit all three recommender denoising jobs in parallel.

mkdir -p /d/hpc/home/sm79111/thesis/results/deep_ksvd

JOB1=$(sbatch --time=12:00:00 --parsable run_deep_ksvd_recommender_denoising_1.sh 0.05)
JOB2=$(sbatch --time=12:00:00 --parsable run_deep_ksvd_recommender_denoising_1.sh 0.10)
JOB3=$(sbatch --time=12:00:00 --parsable run_deep_ksvd_recommender_denoising_1.sh 0.20)

echo "Submitted:"
echo "  p=0.05 → job $JOB1"
echo "  p=0.10 → job $JOB2"
echo "  p=0.20 → job $JOB3"
echo ""
echo "Monitor : squeue -u \$USER"
echo "Cancel  : scancel $JOB1 $JOB2 $JOB3"
