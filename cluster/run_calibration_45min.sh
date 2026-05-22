#!/bin/bash
#SBATCH --job-name=run_cal
#SBATCH --account=ag_inres_luedeling
#SBATCH --partition=intelsr_short
#SBATCH --output=/home/lcaspers_hpc/logs/run_cal%A_%a.out
#SBATCH --error=/home/lcaspers_hpc/logs/run_cal%A_%a.err
#SBATCH --array=1-119
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:45:00

# Load R module if needed on your cluster
module load R
module load libsodium
#module load Arrow

# Run R script with array ID as argument
# Replace 'your_script.R' with your actual R script name
Rscript /home/lcaspers_hpc/code/calibration_hierach_model/run_cluster.R --job-id=${SLURM_ARRAY_TASK_ID}