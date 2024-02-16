#!/bin/bash -l
#SBATCH --cpus-per-task=32                 # CORES per task
#SBATCH --qos=default                      # SLURM qos
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --time=00:15:00                    # time (HH:MM:SS)
#SBATCH --partition=cpu                    # partition
#SBATCH --account=p200301                  # project account<200b>

PROGRAM_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/test/build/mainOmp"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" "$PROGRAM_PATH"