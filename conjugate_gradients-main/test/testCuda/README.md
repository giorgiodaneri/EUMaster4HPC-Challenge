In order to test the accelerated CUDA CG solver, use the interactive node
``` bash
salloc -A p200301 --res gpudev -q dev -N 1 -t 00:30:00
```

First of all you need to load the modules necessary for program compilation and execution
``` bash
module load CUDA 
```

Then compile using the CUDA compiler nvcc with:

``` bash
make
```

Then launch a batch job contaning the **run.sh** slurm batch script with: 

``` bash
sbatch run.sh
```