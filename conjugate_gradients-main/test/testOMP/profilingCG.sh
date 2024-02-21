#!/bin/bash -l
#SBATCH --cpus-per-task=128                # CORES per task
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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES="threads"
srun --cpus-per-task=$SLURM_CPUS_PER_TASK vtune -collect hotspots -result-dir profilerData build/mainOmp ../../io/matrix.bin ../../io/rhs.bin ../../io/sol.bin
