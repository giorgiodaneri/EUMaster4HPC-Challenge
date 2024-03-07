#!/bin/bash -l
#SBATCH --cpus-per-task=1                                            # Set the number of CORES per task
#SBATCH --qos=default                                                # SLURM qos
#SBATCH --nodes=2                                                    # number of nodes
#SBATCH --ntasks=2                                                   # number of tasks
#SBATCH --ntasks-per-node=1                                          # number of tasks per node
#SBATCH --time=01:00:00                                              # time (HH:MM:SS)
#SBATCH --partition=cpu                                              # partition
#SBATCH --account=p200301                                            # project account

# Correctly setting BUILD_DIR to the directory containing the executable
BUILD_DIR="/home/users/u101381/EUMaster4HPC-Challenge/conjugate_gradients-main/build/test/testMPI"

module load CMake OpenMPI 

rm -r build
mkdir build
cd build

cmake .. -DBUILD_MPI=ON -DDEBUG=ON
make
echo "--------------------------------------------------------------------------------"

srun $BUILD_DIR/mainMPI_OpenMP /home/users/u101381/EUMaster4HPC-Challenge/conjugate_gradients-main/io/matrix.bin /home/users/u101381/EUMaster4HPC-Challenge/conjugate_gradients-main/io/rhs.bin /home/users/u101381/EUMaster4HPC-Challenge/conjugate_gradients-main/io/sol.bin 10

