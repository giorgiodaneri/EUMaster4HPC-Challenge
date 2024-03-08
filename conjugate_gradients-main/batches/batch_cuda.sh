#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --gpus-per-task=1                  # number of gpu per task
#SBATCH --cpus-per-task=1                  # number of cores per task
#SBATCH --time=00:01:00                    # time (HH:MM:SS)
#SBATCH --partition=gpu                    # partition
#SBATCH --account=p200301                  # project account
#SBATCH --qos=default                      # SLURM qos

mkdir -p ../results_cuda

# Load CUDA module
module load CUDA

# Change directory
make ../test/testCuda

SAMPLE_PY_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/sample.py"
PROGRAM_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/test/testCuda/main"

chmod +rx "$SAMPLE_PY_PATH"

# Number of samples to take
NUM_SAMPLES=10

srun "$SAMPLE_PY_PATH" $NUM_SAMPLES "$PROGRAM_PATH" ../io/matrix.bin ../io/rhs.bin ../io/sol.bin