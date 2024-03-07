#!/bin/bash -l

# SLURM directives
#SBATCH --cpus-per-task=1                  # CORES per task
#SBATCH --qos=default                      # SLURM qos
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --time=10:00:00                    # time (HH:MM:SS)
#SBATCH --partition=cpu                    # partition
#SBATCH --account=p200301                  # project account

# Load necessary module
module load intel

# Set variables
PROGRAM="random_spd_system"
LINKS="-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
FLAGS="-I${MKLROOT}/include"

# Check if random_spd_system needs to be compiled
if ! command -v $PROGRAM >/dev/null 2>&1; then
    echo "$PROGRAM is not found. Compiling..."
    icpx -O2 ${FLAGS} src/${PROGRAM}.cpp -o ${PROGRAM} $LINKS
else
    echo "$PROGRAM is already present. Skipping compilation."
fi


mkdir -p io/4x4
srun ./$PROGRAM 4 io/4x4/matrix.bin io/4x4/rhs.bin

mkdir -p io/1000x1000
srun ./$PROGRAM 1000 io/1000x1000/matrix.bin io/1000x1000/rhs.bin

mkdir -p io/10000x10000
srun ./$PROGRAM 10000 io/10000x10000/matrix.bin io/10000x10000/rhs.bin

mkdir -p io/30000x30000
srun ./$PROGRAM 30000 io/30000x30000/matrix.bin io/30000x30000/rhs.bin
