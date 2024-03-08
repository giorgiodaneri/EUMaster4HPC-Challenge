# Accelerated Conjugate Gradient method
Project developed as part of a programming challenge for the **EuroHPC Summit** 2024. The objective is to accelerate the well-known iterative solver by utilizing different programming parallel languages. Our implementation features single node as well as multi-node versions, with the aim of exploiting the power of heterogeneous computing. 
The contributors are students of the **EUMaster4HPC** double degree program in High Performance Computing Engineering: \
Giorgio Daneri giorgio.daneri@mail.polimi.it \
Lorenzo Migliari lorenzo.migliari.001@student.uni.lu \
Emanuele Bellini emanuele.bellini@mail.polimi.it 

The algorithm has been tested on the **Meluxina supercomputer**, which is ranked #71st as of November 2023.

## Description of the project

The main program in this project is `conjugate_gradients`, which solves the system. It loads and input dense matrix in row-major format and a right-hand-side from given binary files, performs the conjugate gradient iterations until convergence, and then writes the found solution to a given output binary file. A symmetric positive definite matrix and a right-hand-side can be generated using the `random_spd_system.sh` script and program. A detailed description of the project, the approach we used, the testing and profiling of the code can be found in the report. Now a brief description of the steps to make it work.

First of all you need to load the module necessary for program compilation and execution
``` bash
module load intel 
```

Create a directory for the input and output files
``` bash
mkdir io
```

Then generate a random SPD system of n equations and unknowns (e.g. 10000)
``` bash
./random_spd_system.sh 10000 io/matrix.bin io/rhs.bin
```

## Implementations
Our code features several implementations, which leverage different programming languages and optimized linear algebra libraries:
- openMP on a single CPU node
- openMP + MPI hybrid to split the workload among different nodes 
- openACC offloading of the gemv kernel to the GPU, single node. Integration of Intel MKL library
- CUDA accelerator for single node GPU + CuBLAS library
- SYCL FPGA implementation + oneMKL library
    
## Testing
If you dont' want to bother with the process of compiling the programs in an interactive node, we set up some batch files scripts for automatic testing and performance evaluation. You can find them in the [batches folder](/conjugate_gradients-main/batches/) and they can be used in the login node. There are batch files for each implementation (i.e. batch_cuda). You can edit it to set the number of samples to be executed. 

``` bash
# Number of samples to take
NUM_SAMPLES=10
```

Bear in mind that you should first generate the system by following the above commands. Then launch the script with

``` bash
sbatch batch_cuda.sh
```

If you'd like to use the interactive node and compile the program yourself, head over to the [test folder](/conjugate_gradients-main/test/) and choose one of the directiores corresponding to these implementations to run some tests. You will find a README in each one, just follow the instructions to compile and execute the code.

## Compilation with Cmake 

Available for:
- openMP version.
- openMP + MPI hybrid version.
- openACC version.

For the compilation of the FPGA and CUDA version head over to the [test folder](/conjugate_gradients-main/test/) and follow the instructions there.

### Configuration Options:

- `DEBUG_MODE`: Toggle debug mode on or off. This option can help with debugging. The default setting is `OFF`.
    - Usage: `-DDEBUG_MODE=ON`

- `BUILD_MPI`: Enable the building of the MPI version.
    - Usage: `-DBUILD_MPI=ON`
    - Modules needed: `module load OpenMPI`

- `BUILD_OMP`: Enable the building of the OpenMP version.
    - Usage: `-DBUILD_OMP=ON`
    - Modules needed: `module load OpenBLAS`

- `BUILD_OPENACC`: Enable the building of the OpenACC version.
    - Usage: `-DBUILD_OPENACC=ON`
    - Modules needed: `module load intel`

- `BUILD_SERIAL`: Enable the building of the serial version. 
    - Usage: `-DBUILD_SERIAL=ON -DCMAKE_CXX_COMPILER=icpx`.  Bear in mind that if you compile using this command you are setting the environment variable CXX_COMPILER to a different value. This may conflict with other builds. If you have problems while compiling the other implementations, be sure to add `-DCMAKE_CXX_COMPILER=gcc` when compiling to reset it. 
    - Modules needed: `module load intel` 

As for the CUDA implementation, you need to follow a different way since we didn't configure a CMakeFile for it. Please head over to [this folder](/conjugate_gradients-main/test/testCuda/) and follow the instructions.

### Example Command

To build the project with debug mode enabled and MPI support, navigate to the project's directory and run the following `cmake` command:

``` bash
    mkdir build
    cmake -S /path/to/source -B build -DDEBUG_MODE=ON -DBUILD_MPI=ON
    cd build
    make
```

Adjust the options as necessary to fit your build requirements. 
