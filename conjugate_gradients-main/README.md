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
    
To compile the program, head over to the [test folder](/conjugate_gradients-main/test/) and choose one of the directiores corresponding to these implementations to run some tests. You will find a README in each one, just follow the instructions to compile and execute the code.

Finally, we set up some batch file scripts for automatic testing and performance evaluation. You can find some for CUDA [here](/conjugate_gradients-main/batch_cuda/), for OpenMP in [batches](/conjugate_gradients-main/batches/) and finally for OpenACC [here](/conjugate_gradients-main/batch_openACC/).

