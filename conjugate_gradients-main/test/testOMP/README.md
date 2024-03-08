In order to test the optimized serial code, use the interactive node
``` bash
salloc -A p200301 --res cpudev -q dev -N 1 -t 00:30:00
```

First of all you need to load the modules necessary for program compilation and execution
``` bash
module load CMake OpenBLAS
```

To compile the program, create a new directory that will contain all the build files:

``` bash
mkdir build
cd build
```
Then compile using GCC with:

``` bash
cmake ..
make
```

You can configure the execution parameters in the **runCG.sh** slurm batch script, particularly the `--cpus-per-task` parameter, which is the number of openMP threads.

``` sh
#!/bin/bash -l
#SBATCH --cpus-per-task=32                 # CORES per task
#SBATCH --qos=default                      # SLURM qos
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --time=00:15:00                    # time (HH:MM:SS)
#SBATCH --partition=cpu                    # partition
#SBATCH --account=p200301                  # project account<200b>

PROGRAM_PATH="$HOME/EUMaster4HPC-Challenge/conjugate_gradients-main/test/testOMP/build/mainOmp"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export OMP_PROC_BIND=close
#export OMP_PLACES="threads"
#export GOMP_CPU_AFFINITY="0-32, 64-96"
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" "$PROGRAM_PATH" ../../io/matrix.bin ../../io/rhs.bin ../../io/sol.bin
```
Finally, launch the job with:

``` bash
sbatch runCG.sh
```

You can check the job status with 

``` bash
watch squeue --me
```

Once the job has completed the execution, check the results with

``` bash
cat slurm-file-id
```

If you'd like more information on the code performance and to do some analysis, we have set up the **VTune Intel profiler** for you. Load the corresponding module
``` bash
module load VTune
```

Then run the program with another bash script called **profilerCG.sh**, which launches the profiler together with the application
``` bash
sbatch profilerCG.sh
```