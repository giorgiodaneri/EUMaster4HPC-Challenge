In order to test the accelerated openACC CG solver, use the interactive node
``` bash
salloc -A p200301 --res gpudev -q dev -N 1 -t 00:30:00
```

First of all you need to load the modules necessary for the Intel OneAPI MKL library:
``` bash
module load intel
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

Then launch a batch job contaning the **accRunCG.sh** slurm batch script with: 

``` bash
sbatch accRunCG.sh
```