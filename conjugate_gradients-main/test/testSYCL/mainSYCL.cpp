#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <CL/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <iostream>



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


double dot(const double * x, const double * y, size_t size)
{
    double result = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}

void dot_device(sycl::queue& queue, const double* x_dev, const double* y_dev, size_t size, double* result_dev) {

    oneapi::mkl::blas::dot(queue, size, x_dev, 1, y_dev, 1, result_dev).wait();
}

void axpby(double alpha, const double * x, double beta, double * y, size_t size)
{
    // y = alpha * x + beta * y

    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

void axpby_device(sycl::queue& queue, double alpha, const double* x_dev, double* beta, double* y_dev, size_t size) {
    
    if (*beta != 1.0) {
        oneapi::mkl::blas::scal(queue, size, *beta, y_dev, 1).wait();
    }

    oneapi::mkl::blas::axpy(queue, size, alpha, x_dev, 1, y_dev, 1).wait();
}

void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols)
{
    // y = alpha * A * x + beta * y;

    for(size_t r = 0; r < num_rows; r++)
    {
        double y_val = 0.0;
        for(size_t c = 0; c < num_cols; c++)
        {
            y_val += alpha * A[r * num_cols + c] * x[c];
        }
        y[r] = beta * y[r] + y_val;
    }
}

void gemv_device(sycl::queue& queue, double alpha, const double* A_dev, const double* x_dev, double beta, double* y_dev, size_t num_rows, size_t num_cols) {
    oneapi::mkl::blas::row_major::gemv(queue,
                                       oneapi::mkl::transpose::nontrans,
                                       num_rows, num_cols,
                                       alpha,
                                       A_dev, num_cols,
                                       x_dev, 1,
                                       beta,
                                       y_dev, 1).wait();
}


void conjugate_gradients(sycl::queue &queue, const double *A_host, const double *b_host, double *x_host, size_t size, int max_iters, double rel_error)
{
    double *A_dev = sycl::malloc_device<double>(size * size, queue);
    queue.memcpy(A_dev, A_host, sizeof(double) * size * size).wait();

    double *b_dev = sycl::malloc_device<double>(size, queue);
    queue.memcpy(b_dev, b_host, sizeof(double) * size).wait();

    double *r_dev = sycl::malloc_device<double>(size, queue);
    queue.memcpy(r_dev, b_host, sizeof(double) * size).wait();

    double *p_dev = sycl::malloc_device<double>(size, queue);
    queue.memcpy(p_dev, b_host, sizeof(double) * size).wait();

    double *x_dev = sycl::malloc_device<double>(size, queue);
    queue.memset(x_dev, 0, size * sizeof(double)).wait();

    double *Ap_dev = sycl::malloc_device<double>(size, queue);
    double* rr_new_dev = sycl::malloc_device<double>(1, queue);
    double* beta_dev = sycl::malloc_device<double>(1, queue);

    double* bb_shared = sycl::malloc_shared<double>(1, queue);
    double* rr_shared = sycl::malloc_shared<double>(1, queue);
    double* alpha_shared = sycl::malloc_shared<double>(1, queue);
    double* pAp_dot_shared = sycl::malloc_shared<double>(1, queue);
    double* error_shared = sycl::malloc_shared<double>(1, queue);

    // bb = dot(b, b, size);
    dot_device(queue, b_dev, b_dev, size, bb_shared);
    
    // rr = bb;
    *rr_shared = *bb_shared;

    queue.wait();

    int num_iters;

    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        gemv_device(queue, 1.0, A_dev, p_dev, 0.0, Ap_dev, size, size);
        dot_device(queue, p_dev, Ap_dev, size, pAp_dot_shared);

        *alpha_shared  = *rr_shared / *pAp_dot_shared; // Calculate alpha (maybe perform in the kernel)
        queue.wait();

        queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class axpby_kernel>(sycl::range<1>(size), [=](sycl::id<1> idx) {
            int i = idx[0];
            // Update r_dev in parallel: r_dev = neg_alpha * Ap_dev + 1.0 * r_dev
            r_dev[i] = -(*alpha_shared) * Ap_dev[i] + r_dev[i];
            // Update x_dev in parallel: x_dev = alpha * p_dev + 1.0 * x_dev
            x_dev[i] = *alpha_shared * p_dev[i] + x_dev[i];
        });
        }).wait();

        dot_device(queue, r_dev, r_dev, size, rr_new_dev);

        queue.submit([&](sycl::handler& cgh) {
            cgh.single_task([=]() {

                *beta_dev = *rr_new_dev / *rr_shared; // Update beta_dev
                *rr_shared = *rr_new_dev; // Update rr_dev for the next iteration
                *error_shared = std::sqrt(*rr_shared / *bb_shared);

            });
        }).wait(); 

        if (*error_shared < rel_error)
            break; 

        axpby_device(queue, 1.0, r_dev, beta_dev, p_dev, size); //TODO problea con l'1.0
    }

    // debugging
    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(*rr_shared / *bb_shared));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(*rr_shared / *bb_shared));
    }


    sycl::free(A_dev, queue);
    sycl::free(b_dev, queue);
    sycl::free(x_dev, queue);
    sycl::free(r_dev, queue);
    sycl::free(p_dev, queue);
    sycl::free(Ap_dev, queue);
    sycl::free(beta_dev, queue);
    sycl::free(rr_shared, queue);
    sycl::free(bb_shared, queue);
    sycl::free(alpha_shared, queue);
    sycl::free(error_shared, queue);
    sycl::free(pAp_dot_shared, queue);
}





int main(int argc, char ** argv)
{
    printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    printf("All parameters are optional and have default values\n");
    printf("\n");

    const char * input_file_matrix = "io/matrix.bin";
    const char * input_file_rhs = "io/rhs.bin";
    const char * output_file_sol = "io/sol.bin";
    int max_iters = 1000;
    double rel_error = 1e-9;

    auto selector = sycl::default_selector{};
    sycl::queue queue(selector);

    std::cout << "Running on " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

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



    double * matrix;
    double * rhs;
    size_t size;

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
    double * sol = new double[size];
    conjugate_gradients(queue, matrix, rhs, sol, size, max_iters, rel_error);
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

    return 0;
}