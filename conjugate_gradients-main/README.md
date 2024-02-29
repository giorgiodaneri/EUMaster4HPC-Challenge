# Please follow the steps for running the code on MeluXina

# Conjugate gradient method



## Intruduction

Hello, I am John Doe, and I am a student on the University of Science. This is my final project in the programming class.

It is a program which solves a system of equations using the conjugate gradient method. It is an iterative solver, which needs only matrix-vector multiplications and some vector operations to find a solution. To find more information about the algorithm, read your notes from past Linear algebra classes, or see e.g. [this Wikipedia page](https://en.wikipedia.org/wiki/Conjugate_gradient_method).


## Description of the project

The main program in this project is `conjugate_gradients`, which solves the system. It loads and input dense matrix in row-major format and a right-hand-side from given binary files, performs the conjugate gradient iterations until convergence, and then writes the found solution to a given output binary file. A symmetric positive definite matrix and a right-hand-side can be generated using the `random_spd_system.sh` script and program.

In order to test your code on MeluXina, please use the interactive node (for quick checking)
``` bash
salloc -A p200301 --res cpudev -q dev -N 1 -t 00:30:00
```

First of all you need to load the modules necessary for program compilation and execution
``` bash
module load intel OpenMPI CMake
```

Then generate a random SPD system of n equations and unknowns (e.g. 10000)
``` bash
./random_spd_system.sh 10000 io/matrix.bin io/rhs.bin
```

Create a directory for the input and output files

``` bash
mkdir io
```

To compile the program, head over to the [test folder](/conjugate_gradients-main/test/) and create a new directory:

``` bash
mkdir build
cd build
```
Then compile with:

``` bash
cmake ..
make
# if you want to make compilation faster, use -j flag followed by the number of files to compile
# e.g. make -j 2
```

Then head back to [main folder](/conjugate_gradients-main/) and configure the parameters in the **runCG.sh** slurm batch script, particularly the --cpus-per-task parameter, which is the number of openMP threads. If you want to run the program sequentially, just do `--cpus-per-task=1`.

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
#iNumber of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --cpus-per-task=$SLURM_CPUS_PER_TASK ./conjugate_gradients
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

