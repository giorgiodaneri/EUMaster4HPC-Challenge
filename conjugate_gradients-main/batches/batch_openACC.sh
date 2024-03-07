#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --gpus-per-task=1                  # number of gpu per task
#SBATCH --cpus-per-task=1                  # number of cores per task
#SBATCH --time=00:15:00                    # time (HH:MM:SS)
#SBATCH --partition=gpu                    # partition
#SBATCH --account=p200301                  # project account
#SBATCH --qos=default                      # SLURM qos

mkdir -p ../results_openACC

module load CMake intel

cd ../test/testOpenACC

mkdir build

cd build

cmake ..

make

cd ../../../batches

SAMPLE_PY_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/sample.py"
PROGRAM_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/test/testOpenACC/build/main"

chmod +rx "$SAMPLE_PY_PATH"

# Number of samples to take
NUM_SAMPLES=5

srun "$SAMPLE_PY_PATH" $NUM_SAMPLES "$PROGRAM_PATH" ../io/matrix.bin ../io/rhs.bin ../io/sol.bin