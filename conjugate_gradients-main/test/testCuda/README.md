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

If you'd like to profile the application, you can use the powerful tool NVIDIA Nsight. 
``` bash
nsys profile --stats=true main ../../io/matrix.bin ../../io/rhs.bin ../../io/sol.bin
cat slurm_file_id
```