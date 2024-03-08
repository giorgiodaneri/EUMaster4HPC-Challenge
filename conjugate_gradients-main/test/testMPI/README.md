# Manual Compilation Instructions for MPI and OpenMP (Release Version)
## Step 1: Define Compilation Flags

Set the compilation flags for Release build optimizations and support for both MPI and OpenMP. Adjust these flags as necessary for your specific system and compiler.

    CXXFLAGS="-O3 -Wall -march=native -DOMPI_SKIP_MPICXX -fopenmp"
    MPI_LIB_PATH="/path/to/mpi/lib"

Replace /path/to/mpi/lib with the actual path to your MPI installation library directory.

## Step 2: Compile the Main Program

Compile the main program using the MPI and OpenMP flags. Ensure you link against the MPI and OpenMP libraries correctly.

    mpicxx $CXXFLAGS -o mainMPI_OpenMP mainMPI.cpp -L$MPI_LIB_PATH -lmpi -lopenmp

This command uses mpicxx as the MPI compiler wrapper, which automatically includes the necessary MPI compilation and linking flags.

## Step 3: Run the Program

With the executable compiled, you can run your program across multiple processes using mpirun or another MPI job launcher depending on your MPI implementation:

    mpirun -np 4 ./mainMPI_OpenMP

This example launches the program with 4 processes. Adjust the number of processes (-np) as required.
