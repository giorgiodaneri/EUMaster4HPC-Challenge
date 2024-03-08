#!/bin/bash -l
#SBATCH --qos=default                      # SLURM qos
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --time=01:00:00                    # time (HH:MM:SS)
#SBATCH --partition=fpga                   # partition
#SBATCH --account=p200301                  # project account
#SBATCH --cpus-per-task=1                  # number of cores per task

module load ifpgasdk 520nmx intel env/release/latest ifpga/2021.3.0 intel-compilers
module load env/release/2023.1 imkl/2023.1.0

PROGRAM_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/test/testSYCL/mainSYCL.cpp"
EXECUTABLE_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/test/testSYCL/"
EXECUTABLE_NAME="main.fpga_emu"
MATRIX_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/io/matrix.bin"
RHS_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/io/rhs.bin"
SOLUTION_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/io/sol.bin"
SAMPLE_PY_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/sample.py"

# Number of samples to take
NUM_SAMPLES=3

# Compile the program
icpx -fsycl -DMKL_ILP64 -I${MKLROOT}/include "$PROGRAM_PATH" -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -ltbb -pthread -ldl -lm -o "${EXECUTABLE_PATH}${EXECUTABLE_NAME}"

echo "--------------------------------------------------------------------------------"

srun "$SAMPLE_PY_PATH" $NUM_SAMPLES "${EXECUTABLE_PATH}${EXECUTABLE_NAME}" "$MATRIX_PATH" "$RHS_PATH" "$SOLUTION_PATH"
