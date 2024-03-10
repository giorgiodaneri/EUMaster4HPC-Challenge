#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <chrono>
#include <mpi.h>
#include <stdint.h>
#include <limits.h>

// Define the macro based on SIZE_MAX
// Use MPI_SIZE_T for size_t in MPI
#if SIZE_MAX == UCHAR_MAX
#define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "something is not right"
#endif

bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out)
{
    double * matrix;
    size_t num_rows;
    size_t num_cols;

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    matrix = new double[num_rows * num_cols];
    fread(matrix, sizeof(double), num_rows * num_cols, file);

    *matrix_out = matrix;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;

    fclose(file);

    return true;
}

bool write_matrix_to_file(const char * filename, const double * matrix, size_t num_rows, size_t num_cols)
{
    FILE * file = fopen(filename, "wb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fwrite(&num_rows, sizeof(size_t), 1, file);
    fwrite(&num_cols, sizeof(size_t), 1, file);
    fwrite(matrix, sizeof(double), num_rows * num_cols, file);

    fclose(file);

    return true;
}

void print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file = stdout)
{
    fprintf(file, "%zu %zu\n", num_rows, num_cols);
    for(size_t r = 0; r < num_rows; r++)
    {
        for(size_t c = 0; c < num_cols; c++)
        {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}

#pragma omp declare simd
double dot(const double *x, const double *y, size_t size)
{
    double result = 0.0;

#pragma omp parallel for simd reduction(+ : result)
    for (size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}

#pragma omp declare simd
void axpby(double alpha, const double *x, double beta, double *y, size_t size)
{
#pragma omp parallel for simd
    for (size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

void precA(double* matrix, double* vector, double* result, int num_rows, int num_col) {
#pragma omp parallel for
    for (int i = 0; i < num_rows; i++) {
        double sum = 0;
#pragma omp simd reduction(+ : sum)
        for (int j = 0; j < num_col; j++) {
            sum += matrix[i * num_col + j] * vector[j];
        }
        result[i] = sum;
    }
}

// Function to calculate partitions of the matrix for scattering and gathering
void calculateMatrixPartition(size_t size, int world_size,
                              int* divide_at_index, int* number_element_per_partition,
                              int* rows_per_process_array, int* row_displacements) {

    int elements_per_row = size;
    int base_rows_per_process = size / world_size; // How many rows each process will get
    int extra_elements = size % world_size; // If the number is not divisible we have a reminder of rows
    int current_element_index = 0;
    int current_row_displacement = 0;

    for (int cur_rank = 0; cur_rank < world_size; cur_rank++) {
        divide_at_index[cur_rank] = current_element_index;
        int rows_for_this_process = base_rows_per_process + (cur_rank < extra_elements ? 1 : 0); // Assign the extra rows to the processes in order
        number_element_per_partition[cur_rank] = rows_for_this_process * elements_per_row;
        rows_per_process_array[cur_rank] = rows_for_this_process; // Store the number of rows this process handles
        row_displacements[cur_rank] = current_row_displacement; // Displacement in terms of rows

        current_element_index += number_element_per_partition[cur_rank]; // Increment element index for the next process
        current_row_displacement += rows_for_this_process; // Increment row displacement for the next process
    }
}

void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error,
                         int world_size, int rank)
{
    int* divide_at_index = new int[world_size]; // displacements for Scatterv
    int* number_element_per_partition = new int[world_size]; // counts_send for Scatterv
    int* rows_per_process_array = new int[world_size]; // counts_recv for Gatherv
    int* row_displacements = new int[world_size]; // displacements for Gatherv

    calculateMatrixPartition(size, world_size,
                             divide_at_index, number_element_per_partition, rows_per_process_array, row_displacements);

    double* local_matrix = new double[number_element_per_partition[rank]]; // Each process has a portion of the matrix
    double* local_result = new double[rows_per_process_array[rank]]; // To store the result of the matrix vector multiplication
    double* p = new double[size]; // Each process has the entire vector
    double* Ap = nullptr; // Final result of the vector matrix multiplication
    bool continueLoop = true; // To make all processes exit the loop
    double* r = nullptr; 
    double alpha, beta, bb, rr, rr_new;
    int num_iters;

    // Divide the matrix among the processes
    MPI_Scatterv(A, number_element_per_partition, divide_at_index, MPI_DOUBLE, local_matrix,
                 number_element_per_partition[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        r = new double[size];
        Ap = new double[size];

        for (size_t i = 0; i < size; i++) {
            x[i] = 0.0;
            r[i] = b[i];
            p[i] = b[i];
        }

        bb = dot(b, b, size);
        rr = bb;
    }

    for(num_iters = 1; num_iters <= max_iters && continueLoop; num_iters++)
    {   
        // Broadcast the new vector
        MPI_Bcast(p, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Perform matrix vector multiplication on each portion
        precA(local_matrix, p, local_result, rows_per_process_array[rank], size);
        
        // Gather the result of the multiplication in Ap (only on process 0)
        MPI_Gatherv(local_result, rows_per_process_array[rank], MPI_DOUBLE, Ap,
                    rows_per_process_array, row_displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Perform other calculation on process 0
        if(rank == 0) {
            alpha = rr / dot(p, Ap, size);
            axpby(alpha, p, 1.0, x, size);
            axpby(-alpha, Ap, 1.0, r, size);
            rr_new = dot(r, r, size);
            beta = rr_new / rr; 
            rr = rr_new;

            if(std::sqrt(rr / bb) < rel_error) {
                continueLoop = false; // Prepare to exit the loop
            }
        }

        // Broadcast the decision to all processes
        MPI_Bcast(&continueLoop, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD); 
        
        if(!continueLoop) {
            break; // All processes exit the loop
        }

        if(rank == 0) {
            axpby(1.0, r, beta, p, size);
        }
    }

    delete[] p;
    delete[] divide_at_index;
    delete[] number_element_per_partition;
    delete[] rows_per_process_array;
    delete[] row_displacements;

    if(rank == 0) {
        delete[] r;
        delete[] Ap;

        if(num_iters <= max_iters)
        {
            printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
        }
        else
        {
            printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
        }
    }

}

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double * matrix = nullptr;
    double * rhs = nullptr;
    double * sol = nullptr;
    size_t size;
    const char * input_file_matrix = nullptr;
    const char * input_file_rhs = nullptr;
    const char * output_file_sol =  nullptr;
    int max_iters = 1000;
    double rel_error = 1e-9;

    if(rank == 0) {

        printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
        printf("All parameters are optional and have default values\n");
        printf("\n");

        input_file_matrix = "io/matrix.bin";
        input_file_rhs = "io/rhs.bin";
        output_file_sol = "io/sol.bin";
        
        if(argc > 1) input_file_matrix = argv[1];
        if(argc > 2) input_file_rhs = argv[2];
        if(argc > 3) output_file_sol = argv[3];
        if(argc > 4) max_iters = atoi(argv[4]);
        if(argc > 5) rel_error = atof(argv[5]);

        printf("Command line arguments:\n");
        printf("  input_file_matrix: %s\n", input_file_matrix);
        printf("  input_file_rhs:    %s\n", input_file_rhs);
        printf("  output_file_sol:   %s\n", output_file_sol);
        printf("  max_iters:         %d\n", max_iters);
        printf("  rel_error:         %e\n", rel_error);
        printf("\n");

        {
            printf("Reading matrix from file ...\n");
            size_t matrix_rows;
            size_t matrix_cols;
            bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows, &matrix_cols);
            if(!success_read_matrix)
            {
                fprintf(stderr, "Failed to read matrix\n");
                return 1;
            }
            printf("Done\n");
            printf("\n");

            printf("Reading right hand side from file ...\n");
            size_t rhs_rows;
            size_t rhs_cols;
            bool success_read_rhs = read_matrix_from_file(input_file_rhs, &rhs, &rhs_rows, &rhs_cols);
            if(!success_read_rhs)
            {
                fprintf(stderr, "Failed to read right hand side\n");
                return 2;
            }
            printf("Done\n");
            printf("\n");

            if(matrix_rows != matrix_cols)
            {
                fprintf(stderr, "Matrix has to be square\n");
                return 3;
            }
            if(rhs_rows != matrix_rows)
            {
                fprintf(stderr, "Size of right hand side does not match the matrix\n");
                return 4;
            }
            if(rhs_cols != 1)
            {
                fprintf(stderr, "Right hand side has to have just a single column\n");
                return 5;
            }

            size = matrix_rows;
        }

        printf("Solving the system ...\n");
        sol = new double[size];
    }

    MPI_Bcast(&size, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    conjugate_gradients(matrix, rhs, sol, size, max_iters, rel_error, world_size, rank);

    if(rank == 0) {

        printf("Done\n");
        printf("\n");

        printf("Writing solution to file ...\n");
        bool success_write_sol = write_matrix_to_file(output_file_sol, sol, size, 1);
        if(!success_write_sol)
        {
            fprintf(stderr, "Failed to save solution\n");
            return 6;
        }
        printf("Done\n");
        printf("\n");

        delete[] matrix;
        delete[] rhs;
        delete[] sol;

        printf("Finished successfully\n");
    }

    MPI_Finalize();
    return 0;
}
