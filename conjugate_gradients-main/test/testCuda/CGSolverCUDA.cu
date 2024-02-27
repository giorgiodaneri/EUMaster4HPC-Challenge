#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <chrono>
#include <time.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

extern "C++"
{
#include "../../include/CGSolver.hpp"
#include "../../include/CGSolverCuda.hpp"
}

// vector vector multiply
#define BLOCK_SIZE 256
#define SIZE 5000 // TODO: handle this better, maybe pass it as a parameter to the functions

// __global__ void vecVecMult(double *a, double *b, double *c)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < SIZE)
//     {
//         c[i] = a[i] * b[i];
//     }
// }

// vector vector add
__global__ void vecVecAdd(double *a, double *b, double *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SIZE)
    {
        c[i] = a[i] + b[i];
    }
}

// vector vector subtract
__global__ void vecVecSub(double *a, double *b, double *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SIZE)
    {
        c[i] = a[i] - b[i];
    }
}

// matrix vector multiply
__global__ void matVecMult(double *A, double *x, double *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SIZE)
    {
        double sum = 0;
        for (int j = 0; j < SIZE; j++)
        {
            sum += A[i * SIZE + j] * x[j];
        }
        y[i] = sum;
    }
}

// __global__ void matVecMult(double *A, double *b, double *out)
// {
//     __shared__ float b_shared[BLOCK_SIZE];

//     int effective_block_width;
//     if ((blockIdx.x + 1) * BLOCK_SIZE <= SIZE)
//     {
//         effective_block_width = BLOCK_SIZE;
//     }
//     else
//     {
//         // needed to avoid overflow in next row
//         effective_block_width = SIZE % BLOCK_SIZE;
//     }

//     if (threadIdx.x < effective_block_width)
//         b_shared[threadIdx.x] = b[blockIdx.x * BLOCK_SIZE + threadIdx.x];

//     __syncthreads();

//     int idy = blockIdx.y * BLOCK_SIZE + threadIdx.x;
//     float tmp_scal = 0.0;
//     // threads outside matrix dimension are not needed (vertical)
//     if (idy < SIZE)
//     {
//         for (int i = 0; i < effective_block_width; i++)
//         {
//             // take advantage of symmetric matrix for coalesced memory access
//             // tmp_scal += A[idy * SIZE + blockIdx.x * BLOCK_SIZE + i] * b_shared[i];
//             tmp_scal += b_shared[i] * A(blockIdx.x * BLOCK_SIZE + i, idy);
//         }
//         atomicAdd(out + idy, tmp_scal);
//     }
// }

__global__ void dotProduct(double *a, double *b, double *out)
{
    // each block has it's own shared_tmp of size BLOCK_SIZE
    __shared__ double shared_tmp[BLOCK_SIZE];

    // needed for atomicAdd
    if (threadIdx.x + blockDim.x * blockIdx.x == 0)
    {
        *out = 0.0;
    }

    if (blockIdx.x * blockDim.x + threadIdx.x < SIZE)
    {
        shared_tmp[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x] * b[blockIdx.x * blockDim.x + threadIdx.x];
    }
    else
    {
        // needed for the reduction
        shared_tmp[threadIdx.x] = 0.0;
    }

    // reduction within block
    for (int i = blockDim.x / 2; i >= 1; i = i / 2)
    {
        // threads access memory position written by other threads so sync is needed
        __syncthreads();
        if (threadIdx.x < i)
        {
            shared_tmp[threadIdx.x] += shared_tmp[threadIdx.x + i];
        }
    }

    // atomic add the partial reduction in out
    if (threadIdx.x == 0)
    {
        atomicAdd(out, shared_tmp[0]);
    }
}

// scalar vector multiply
__global__ void scalarVecMult(double *a, double *b, double *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SIZE)
    {
        b[i] = a[i] * *c;
    }
}

// divide two scalars
__global__ void divide(double *a, double *b, double *c)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0)
        *c = *a / *b;
}

__global__ void memCopy(double *in, double *out)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < SIZE)
    {
        out[index] = in[index];
    }
}

double dot(const double *x, const double *y, size_t size)
{
    double result = 0.0;
    for (size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}

void axpby(double alpha, const double *x, double beta, double *y, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

// CG solver main function
void solve_cuda(double *A, double *b, double *x, size_t size, int maxIterations, double tolerance)
{
    // define dimension of the grid and block for vectors
    dim3 vecDimBlock(BLOCK_SIZE);
    dim3 vecDimGrid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // define dimension of the grid and block for matrices
    dim3 matDimBlock(BLOCK_SIZE);
    dim3 matDimGrid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    double alpha = 1.0;
    double beta = 0.0;

    // residual
    double *r = new double[size];
    // preconditioned residual
    double *p = new double[size];
    double *Ap = new double[size];
    int num_iters;
    // norm of the residual
    double r_norm = 1.0;
    double b_norm = 1.0;

    // allocate device memory
    double *d_A, *d_b, *d_x, *d_r, *d_p, *d_Ap, *d_temp;
    cudaMalloc((void **)&d_A, size * size * sizeof(double));
    cudaMalloc((void **)&d_b, size * sizeof(double));
    cudaMalloc((void **)&d_x, size * sizeof(double));
    cudaMalloc((void **)&d_r, size * sizeof(double));
    cudaMalloc((void **)&d_p, size * sizeof(double));
    cudaMalloc((void **)&d_Ap, size * sizeof(double));
    cudaMalloc((void **)&d_temp, size * sizeof(double));

    // copy data to device
    cudaMemcpy(d_A, A, size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_r, r, size * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_p, p, size * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_Ap, Ap, size * sizeof(double), cudaMemcpyHostToDevice);

    // allocate memory for five doubles, since cuda needs this type for atomicAdd, rather than double
    // (no overloading exists for variables of type double)
    double *d_alpha, *d_beta, *d_rr, *d_rr_new, *d_bb, *d_temp_scalar;
    cudaMalloc((void **)&d_alpha, sizeof(double));
    cudaMalloc((void **)&d_beta, sizeof(double));
    cudaMalloc((void **)&d_rr, sizeof(double));
    cudaMalloc((void **)&d_rr_new, sizeof(double));
    cudaMalloc((void **)&d_bb, sizeof(double));
    cudaMalloc((void **)&d_temp_scalar, sizeof(double));

    // initialize the residual by copying the rhs into it
    cudaMemcpy(d_r, d_b, size * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_p, d_b, size * sizeof(double), cudaMemcpyDeviceToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // calculate the dot product of the rhs, which is equal to that
    // of the residual
    dotProduct<<<vecDimGrid, vecDimBlock>>>(d_b, d_b, d_bb);
    // copy value of d_bb to d_rr
    cudaMemcpy(d_rr, d_bb, sizeof(double), cudaMemcpyDeviceToDevice);
    // copy value of d_bb (norm of rhs) to host
    cudaMemcpy(&b_norm, d_bb, sizeof(double), cudaMemcpyDeviceToHost);

    // now start the CG solver iterations
    for (num_iters = 0; num_iters < maxIterations; num_iters++)
    {
        // precA(A, p, Ap, size);
        // matVecMult<<<matDimGrid, matDimBlock>>>(d_A, d_p, d_Ap);
        // Perform matrix-vector multiplication using cuBLAS gemv function
        cublasDgemv(handle, CUBLAS_OP_N, size, size, &alpha, d_A, size, d_p, 1, &beta, d_Ap, 1);

        // compute new alpha coefficient
        // alpha = rr / dot(p, Ap, size);
        dotProduct<<<vecDimGrid, vecDimBlock>>>(d_p, d_Ap, d_temp_scalar);
        divide<<<1, 1>>>(d_rr, d_temp_scalar, d_alpha);

        // compute new approximate of the solution at step k+1
        // x_k+1 = x_k + alpha_k * p_k
        // axpby(alpha, p, 1.0, x, size);
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_p, d_temp, d_alpha);
        vecVecAdd<<<vecDimGrid, vecDimBlock>>>(d_x, d_temp, d_x);

        // compute new residual at step k+1
        // r_k+1 = r_k - alpha_k * A * p_k
        // axpby(-alpha, Ap, 1.0, r, size);
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_Ap, d_temp, d_alpha);
        vecVecSub<<<vecDimGrid, vecDimBlock>>>(d_r, d_temp, d_r);

        // update the 2-norm of the residual at step k+1
        // rr_new = dot(r, r, size);
        dotProduct<<<vecDimGrid, vecDimBlock>>>(d_r, d_r, d_rr_new);

        // beta_k = ||r_k+1||^2 / ||r_k||^2
        // beta = rr_new / rr;
        divide<<<1, 1>>>(d_rr_new, d_rr, d_beta);

        // update residual norm
        // rr = rr_new;
        memCopy<<<1, 1>>>(d_rr_new, d_rr);

        // stopping criterion ==> sqrt(||r||^2 / ||b||^2) < rel_error equivalent to 2-norm or euclidean norm
        // need to copy the norm of the residual back to the device
        // in order to evaluate it
        cudaMemcpy(&r_norm, d_rr_new, sizeof(double), cudaMemcpyDeviceToHost);

        if (std::sqrt(r_norm / b_norm) < tolerance)
        {
            break;
        }

        // compute new direction at step k+1
        // p_k+1 = r_k+1 + beta_k * p_k
        // axpby(1.0, r, beta, p, size);
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_p, d_temp, d_beta);
        vecVecAdd<<<vecDimGrid, vecDimBlock>>>(d_r, d_temp, d_p);
    }

    // Record stop event
    cudaEventRecord(stop);
    // Synchronize to ensure that the event recording is completed
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // print the execution time
    printf("Total execution time: %f ms\n", milliseconds);

    // print the relative error and number of iterations
    printf("relative error: %e \n", std::sqrt(r_norm / b_norm));
    printf("number of iterations: %d \n", num_iters);

    // free device memory
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    cudaFree(d_temp);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_rr);
    cudaFree(d_rr_new);
    cudaFree(d_bb);
    cudaFree(d_temp_scalar);

    // Destroy cuBLAS handle
    cublasDestroy(handle);

    // free host memory
    delete[] r;
    delete[] p;
    delete[] Ap;
}

void solve_try(double *A, double *b, double *x, size_t size, int maxIterations, double tolerance)
{
    // define dimension of the grid and block for vectors
    dim3 vecDimBlock(BLOCK_SIZE);
    dim3 vecDimGrid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // define dimension of the grid and block for matrices
    dim3 matDimBlock(BLOCK_SIZE);
    dim3 matDimGrid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // residual
    double beta, bb;
    double alpha, rr;
    // residual
    double *r = new double[size];
    // preconditioned residual
    double *p = new double[size];
    double *Ap = new double[size];
    int num_iters;
    double *temp_scalar = (double *)malloc(sizeof(double));
    double *rr_new = (double *)malloc(sizeof(double));

    for (size_t i = 0; i < size; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    // needed for stopping criterion
    bb = dot(b, b, size);
    rr = bb;

    // allocate device memory
    double *d_A, *d_p, *d_Ap, *d_x, *d_temp, *d_r;
    cudaMalloc((void **)&d_A, size * size * sizeof(double));
    cudaMalloc((void **)&d_p, size * sizeof(double));
    cudaMalloc((void **)&d_Ap, size * sizeof(double));
    cudaMalloc((void **)&d_x, size * sizeof(double));
    cudaMalloc((void **)&d_temp, size * sizeof(double));
    cudaMalloc((void **)&d_r, size * sizeof(double));
    double *d_temp_scalar, *d_alpha, *d_beta, *d_rr, *d_rr_new;
    cudaMalloc((void **)&d_temp_scalar, sizeof(double));
    cudaMalloc((void **)&d_alpha, sizeof(double));
    cudaMalloc((void **)&d_beta, sizeof(double));
    cudaMalloc((void **)&d_rr, sizeof(double));
    cudaMalloc((void **)&d_rr_new, sizeof(double));

    // copy data to device
    cudaMemcpy(d_A, A, size * size * sizeof(double), cudaMemcpyHostToDevice);
    // Create CUDA events for timing
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // now start the CG solver iterations
    for (num_iters = 0; num_iters < maxIterations; num_iters++)
    {
        // precA(A, p, Ap, size);
        // compute matrix vector product
        // first need to get latest value of p from the host and copy it to the device
        cudaMemcpy(d_p, p, size * sizeof(double), cudaMemcpyHostToDevice);
        matVecMult<<<matDimGrid, matDimBlock>>>(d_A, d_p, d_Ap);
        // copy value of d_Ap to Ap
        cudaMemcpy(Ap, d_Ap, size * sizeof(double), cudaMemcpyDeviceToHost);

        // compute new alpha coefficient to guarantee optimal convergence rate
        // copy value of rr to d_rr
        cudaMemcpy(d_rr, &rr, sizeof(double), cudaMemcpyHostToDevice);
        dotProduct<<<vecDimGrid, vecDimBlock>>>(d_p, d_Ap, d_temp_scalar);
        // cudaMemcpy(temp_scalar, d_temp_scalar, sizeof(double), cudaMemcpyDeviceToHost);
        divide<<<1, 1>>>(d_rr, d_temp_scalar, d_alpha);
        // copy value of d_alpha to alpha
        cudaMemcpy(&alpha, d_alpha, sizeof(double), cudaMemcpyDeviceToHost);
        // alpha = rr / *temp_scalar;
        // alpha = rr / dot(p, Ap, size);

        // compute new approximate of the solution at step k+1
        // x_k+1 = x_k + alpha_k * p_k
        // copy value of alpha to d_temp_scalar and x to d_x
        cudaMemcpy(d_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_p, d_temp, d_alpha);
        vecVecAdd<<<vecDimGrid, vecDimBlock>>>(d_x, d_temp, d_x);
        // copy value of d_x to x
        cudaMemcpy(x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost);

        // axpby(alpha, p, 1.0, x, size);

        // compute new residual at step k+1
        // r_k+1 = r_k - alpha_k * A * p_k
        // copy value of r to d_r
        cudaMemcpy(d_r, r, size * sizeof(double), cudaMemcpyHostToDevice);
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_Ap, d_temp, d_alpha);
        vecVecSub<<<vecDimGrid, vecDimBlock>>>(d_r, d_temp, d_r);
        // copy value of d_r to r
        cudaMemcpy(r, d_r, size * sizeof(double), cudaMemcpyDeviceToHost);

        // axpby(-alpha, Ap, 1.0, r, size);

        // compute the 2-norm of the residual at step k+1
        cudaMemcpy(d_r, r, size * sizeof(double), cudaMemcpyHostToDevice);
        dotProduct<<<vecDimGrid, vecDimBlock>>>(d_r, d_r, d_temp_scalar);
        // copy d_rr_new to rr_new
        // USEFUL
        cudaMemcpy(rr_new, d_temp_scalar, sizeof(double), cudaMemcpyDeviceToHost);
        // rr_new = dot(r, r, size);
        // rr_new = *temp_scalar;

        // compute beta coefficient
        // beta_k = ||r_k+1||^2 / ||r_k||^2
        divide<<<1, 1>>>(d_temp_scalar, d_rr, d_beta);
        // copy value of d_beta to beta

        // update residual norm
        memCopy<<<1, 1>>>(d_rr_new, d_rr);
        // copy d_rr to rr
        cudaMemcpy(&rr, d_rr, sizeof(double), cudaMemcpyDeviceToHost);
        // rr = *rr_new;
        // // copy new value of d_rr to rr
        // cudaMemcpy(temp_scalar, d_rr, sizeof(double), cudaMemcpyDeviceToHost);
        // rr = *temp_scalar;
        // cudaMemcpy(d_rr, &rr, sizeof(double), cudaMemcpyHostToDevice);
        // stopping criterion ==> sqrt(||r||^2 / ||b||^2) < rel_error equivalent to 2-norm or euclidean norm
        if (std::sqrt(rr / bb) < tolerance)
        {
            break;
        }

        // compute new direction at step k+1
        // p_k+1 = r_k+1 + beta_k * p_k
        // copy value of r to d_r and beta to
        cudaMemcpy(d_r, r, size * sizeof(double), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_beta, &beta, sizeof(double), cudaMemcpyHostToDevice);
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_p, d_temp, d_beta);
        vecVecAdd<<<vecDimGrid, vecDimBlock>>>(d_r, d_temp, d_p);
        // copy value of d_p to p
        cudaMemcpy(p, d_p, size * sizeof(double), cudaMemcpyDeviceToHost);

        // axpby(1.0, r, beta, p, size);
    }

    // Record stop event
    cudaEventRecord(stop);
    // Synchronize to ensure that the event recording is completed
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // print the execution time
    printf("Total execution time: %f ms\n", milliseconds);

    printf("Relative error is: %e \n", std::sqrt(rr / bb));
    printf("Number of iterations: %d \n", num_iters);
    // free memory allocated on the device
    cudaFree(d_A);
    cudaFree(d_Ap);
    cudaFree(d_p);
    cudaFree(d_x);
    cudaFree(d_temp);
    cudaFree(d_r);
    cudaFree(d_temp_scalar);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_rr);
    cudaFree(d_rr_new);

    delete[] r;
    delete[] p;
    delete[] Ap;
    free(temp_scalar);
    free(rr_new);
}

// main function
void kernel_wrapper(double *matrix, double *rhs, double *sol, size_t size, int max_iters, double rel_error)
{
    printf("Calling kernel!\n");
    solve_cuda(matrix, rhs, sol, size, max_iters, rel_error);
}