#!/bin/bash -l
#SBATCH --cpus-per-task=64                                            # Set the number of CORES per task
#SBATCH --qos=default                                                # SLURM qos
#SBATCH --nodes=1                                                    # number of nodes
#SBATCH --ntasks=1                                                   # number of tasks
#SBATCH --ntasks-per-node=1                                          # number of tasks per node
#SBATCH --time=00:30:00                                              # time (HH:MM:SS)
#SBATCH --partition=cpu                                              # partition
#SBATCH --account=p200301                                            # project account
#SBATCH --output=../result/output_32_%j.txt                           # Output file

#print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
#module purge || print_error_and_exit "No 'module' command"

mkdir -p ../result

SAMPLE_PY_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/sample.py"
PROGRAM_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/test/testOMP/build/mainOmp"

chmod +rx "$SAMPLE_PY_PATH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Number of samples to take
NUM_SAMPLES=3

srun --cpus-per-task=$SLURM_CPUS_PER_TASK "$SAMPLE_PY_PATH" $NUM_SAMPLES "$PROGRAM_PATH" ../io/matrix.bin ../io/rhs.bin ../io/sol.bin
