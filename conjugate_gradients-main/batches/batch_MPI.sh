#!/bin/bash -l
#SBATCH --cpus-per-task=16                                            # Set the number of CORES per task
#SBATCH --qos=default                                                # SLURM qos
#SBATCH --nodes=6                                                   # number of nodes
#SBATCH --ntasks=30                                                   # number of tasks
#SBATCH --ntasks-per-node=5                                        # number of tasks per node
#SBATCH --time=01:00:00                                              # time (HH:MM:SS)
#SBATCH --partition=cpu                                              # partition
#SBATCH --account=p200301                                            # project account


BUILD_DIR="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/build/test/testMPI"
MATRIX_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/io/matrix.bin"
RHS_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/io/rhs.bin"
SOLUTION_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/io/sol.bin"

module load CMake OpenMPI 

mkdir -p ../build
cd ../build

cmake .. -DBUILD_MPI=ON
make

echo "--------------------------------------------------------------------------------"

srun "$BUILD_DIR"/mainMPI_OpenMP "$MATRIX_PATH" "$RHS_PATH" "$SOLUTION_PATH"
