In order to test the optimized serial code, use the interactive node
``` bash
salloc -A p200301 --res cpudev -q dev -N 1 -t 00:30:00
```

First of all you need to load the modules necessary for program compilation and execution
``` bash
module load CMake OpenBLAS
```

To compile the program, head over to the [test folder](/conjugate_gradients-main/test/) and create a new directory:

``` bash
mkdir build
cd build
```
Then compile using the intel icpx compiler for C++ with:

``` bash
cmake ..
make
# if you want to make compilation faster, use -j flag followed by the number of files to compile
# e.g. make -j 2
```

Then launch a batch job contaning the **runCG.sh** slurm batch script with: 

``` bash
sbatch runCG.sh
```