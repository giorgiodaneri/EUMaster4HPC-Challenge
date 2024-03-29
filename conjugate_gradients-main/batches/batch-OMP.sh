#!/bin/bash -l
#SBATCH --cpus-per-task=64                                            # Set the number of CORES per task
#SBATCH --qos=default                                                # SLURM qos
#SBATCH --nodes=1                                                    # number of nodes
#SBATCH --ntasks=1                                                   # number of tasks
#SBATCH --ntasks-per-node=1                                          # number of tasks per node
#SBATCH --time=00:30:00                                              # time (HH:MM:SS)
#SBATCH --partition=cpu                                              # partition
#SBATCH --account=p200301                                            # project account

BUILD_DIR="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/build/test/testOMP"
MATRIX_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/io/matrix.bin"
RHS_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/io/rhs.bin"
SOLUTION_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/io/sol.bin"
SAMPLE_PY_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/sample.py"

module load CMake OpenBLAS

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

rm -r build
mkdir build
cd build

cmake .. -DBUILD_OMP=ON
make

# Number of samples to take
NUM_SAMPLES=10

chmod +rx "$SAMPLE_PY_PATH"

echo "--------------------------------------------------------------------------------"

srun --cpus-per-task="$SLURM_CPUS_PER_TASK" "$SAMPLE_PY_PATH" $NUM_SAMPLES "$BUILD_DIR"/mainOmp "$MATRIX_PATH" "$RHS_PATH" "$SOLUTION_PATH"
