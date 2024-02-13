#!/bin/bash -l
#SBATCH --cpus-per-task=32                # CORES per task
#SBATCH --qos=default                      # SLURM qos
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --time=00:15:00                    # time (HH:MM:SS)
#SBATCH --partition=cpu                    # partition
#SBATCH --account=p200301                  # project account<200b>time 00:05:00

#Check Intel version
vtune --version

#Profile a serial program with VTune
srun --cpus-per-task=$SLURM_CPUS_PER_TASK vtune -collect hotspots -result-dir profilerData test/build/mainOmp