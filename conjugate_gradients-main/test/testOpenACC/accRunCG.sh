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

srun ./test/testOpenACC/build/mainOpenACC ../../io/matrix.bin ../../io/rhs.bin ../../io/sol.bin
