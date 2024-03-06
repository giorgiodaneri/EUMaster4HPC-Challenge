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

#define BLOCK_SIZE 128 // should be best for cublas

// vector vector add
__global__ void vecVecAdd(double *a, double *b, double *c, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        c[i] = a[i] + b[i];
    }
}

// vector vector subtract
__global__ void vecVecSub(double *a, double *b, double *c, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        c[i] = a[i] - b[i];
    }
}

// // NAIVE KERNEL 
// matrix vector multiply
__global__ void matVecMult(double *A, double *x, double *y, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        double sum = 0;
        for (int j = 0; j < size; j++)
        {
            sum += A[i * size + j] * x[j];
        }
        y[i] = sum;
    }
}

// KERNEL 2
// __global__ void matVecMult(double *A, double *b, double *out)
// {
//     __shared__ double b_shared[BLOCK_SIZE];

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
//         b_shared[threadIdx.x] = b[blockIdx.x * blockDim.x + threadIdx.x];

//     __syncthreads();

//     int idy = blockIdx.y * blockDim.y + threadIdx.x;
//     double tmp_scal = 0.0;
//     // threads outside matrix dimension are not needed (vertical)
//     if (idy < SIZE)
//     {
//         for (int i = 0; i < effective_block_width; i++)
//         {
//             // Access A elements using column-major indexing
//             tmp_scal += b_shared[i] * A[(blockIdx.x * blockDim.x + i) * SIZE + idy];
//         }
//         out[idy] = tmp_scal;
//     }
// }

// // GLOBAL MEMORY IMPLEMENTATION
// __global__ void dotProduct(double *a, double *b, double *out) {
// 	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
// 	double tmp = 0.0;
// 	if (index_x == 0) {
// 		for (int i = 0; i < SIZE; i++) {
// 			tmp += b[i] * a[i];
// 		}
// 		*out = tmp;
// 	}
// }

// // SHARED MEMORY IMPLEMENTATION
__global__ void dotProduct(double *a, double *b, double *out, int size)
{
    // each block has it's own shared_tmp of size BLOCK_SIZE
    __shared__ double shared_tmp[BLOCK_SIZE];

    // needed for atomicAdd
    if (threadIdx.x + blockDim.x * blockIdx.x == 0)
    {
        *out = 0.0;
    }

    if (blockIdx.x * blockDim.x + threadIdx.x < size)
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
__global__ void scalarVecMult(double *a, double *b, double *c, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
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

__global__ void memCopy(double *in, double *out, int size)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        out[index] = in[index];  
    }
}

// CG solver main function
void solve_cuda(double *A, double *b, double *x, size_t size, int maxIterations, double tolerance)
{
    // define dimension of the grid and block for vectors
    dim3 vecDimBlock(BLOCK_SIZE);
    dim3 vecDimGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // define dimension of the grid and block for matrices
    dim3 matDimBlock(BLOCK_SIZE);
    dim3 matDimGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);

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
    // cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    // allocate memory for five doubles
    double *d_alpha, *d_beta, *d_rr, *d_rr_new, *d_bb, *d_temp_scalar;
    cudaMalloc((void **)&d_alpha, sizeof(double));
    cudaMalloc((void **)&d_beta, sizeof(double));
    cudaMalloc((void **)&d_rr, sizeof(double));
    cudaMalloc((void **)&d_rr_new, sizeof(double));
    cudaMalloc((void **)&d_bb, sizeof(double));
    cudaMalloc((void **)&d_temp_scalar, sizeof(double));

    // initialize the residual by copying the rhs into it, same for the new direction p
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
    dotProduct<<<vecDimGrid, vecDimBlock>>>(d_b, d_b, d_bb, size);
    // copy value of d_bb to d_rr
    cudaMemcpy(d_rr, d_bb, sizeof(double), cudaMemcpyDeviceToDevice);
    // copy value of d_bb (norm of rhs) to host
    cudaMemcpy(&b_norm, d_bb, sizeof(double), cudaMemcpyDeviceToHost);

    // now start the CG solver iterations
    for (num_iters = 0; num_iters < maxIterations; num_iters++)
    {
        // precA(A, p, Ap, size);
        matVecMult<<<matDimGrid, matDimBlock>>>(d_A, d_p, d_Ap, size);

        // compute new alpha coefficient
        // alpha = rr / dot(p, Ap, size);
        dotProduct<<<vecDimGrid, vecDimBlock>>>(d_p, d_Ap, d_temp_scalar, size);
        divide<<<1, 1>>>(d_rr, d_temp_scalar, d_alpha);

        // compute new approximate of the solution at step k+1
        // x_k+1 = x_k + alpha_k * p_k
        // axpby(alpha, p, 1.0, x, size);
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_p, d_temp, d_alpha, size);
        vecVecAdd<<<vecDimGrid, vecDimBlock>>>(d_x, d_temp, d_x, size);

        // compute new residual at step k+1
        // r_k+1 = r_k - alpha_k * A * p_k
        // axpby(-alpha, Ap, 1.0, r, size);
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_Ap, d_temp, d_alpha, size);
        vecVecSub<<<vecDimGrid, vecDimBlock>>>(d_r, d_temp, d_r, size);

        // update the 2-norm of the residual at step k+1
        // rr_new = dot(r, r, size);
        dotProduct<<<vecDimGrid, vecDimBlock>>>(d_r, d_r, d_rr_new, size);

        // beta_k = ||r_k+1||^2 / ||r_k||^2
        // beta = rr_new / rr;
        divide<<<1, 1>>>(d_rr_new, d_rr, d_beta);

        // update residual norm
        // rr = rr_new;
        memCopy<<<1, 1>>>(d_rr_new, d_rr, size);

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
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_p, d_temp, d_beta, size);
        vecVecAdd<<<vecDimGrid, vecDimBlock>>>(d_r, d_temp, d_p, size);
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
    // cudaDeviceReset();
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
}

void solve_cublas(double *A, double *b, double *x, size_t size, int maxIterations, double tolerance)
{
    // define dimension of the grid and block for vectors
    dim3 vecDimBlock(BLOCK_SIZE);
    dim3 vecDimGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    double alpha = 1.0;
    double beta = 0.0;

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

    // // NEEDED FOR CUBLAS DAXPY:
    // double dd_alpha, dd_beta;
    // // allocate device memory for dd_alpha
    // cudaMalloc((void **)&dd_alpha, sizeof(double));
    // cudaMalloc((void **)&dd_beta, sizeof(double));

    // allocate memory for five doubles
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
    dotProduct<<<vecDimGrid, vecDimBlock>>>(d_b, d_b, d_bb, size);
    // cublasDnrm2(handle, SIZE, d_b, 1, d_bb);
    // copy value of d_bb to d_rr
    cudaMemcpy(d_rr, d_bb, sizeof(double), cudaMemcpyDeviceToDevice);
    // copy value of d_bb (norm of rhs) to host
    cudaMemcpy(&b_norm, d_bb, sizeof(double), cudaMemcpyDeviceToHost);

    // now start the CG solver iterations
    for (num_iters = 0; num_iters < maxIterations; num_iters++)
    {
        // Perform matrix-vector multiplication using cuBLAS gemv function
        cublasDgemv(handle, CUBLAS_OP_T, size, size, &alpha, d_A, size, d_p, 1, &beta, d_Ap, 1);

        // compute new alpha coefficient
        // alpha = rr / dot(p, Ap, size);
        // dotProduct<<<vecDimGrid, vecDimBlock>>>(d_p, d_Ap, d_temp_scalar);
        cublasDdot(handle, size, d_p, 1, d_Ap, 1, d_temp_scalar);
        divide<<<1, 1>>>(d_rr, d_temp_scalar, d_alpha);

        // compute new approximate of the solution at step k+1
        // x_k+1 = x_k + alpha_k * p_k
        // axpby(alpha, p, 1.0, x, size);
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_p, d_temp, d_alpha, size);
        vecVecAdd<<<vecDimGrid, vecDimBlock>>>(d_x, d_temp, d_x, size);
        // copy value of d_alpha to dd_alpha
        // cudaMemcpy(&dd_alpha, d_alpha, sizeof(double), cudaMemcpyDeviceToDevice);
        // cublasDaxpy(handle, SIZE, &dd_alpha, d_p, 1, d_x, 1);

        // compute new residual at step k+1
        // r_k+1 = r_k - alpha_k * A * p_k
        // axpby(-alpha, Ap, 1.0, r, size);
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_Ap, d_temp, d_alpha, size);
        vecVecSub<<<vecDimGrid, vecDimBlock>>>(d_r, d_temp, d_r, size);
        // cublasDaxpy(handle, SIZE, &dd_alpha, d_Ap, 1, d_r, 1);

        // update the 2-norm of the residual at step k+1
        // rr_new = dot(r, r, size);
        dotProduct<<<vecDimGrid, vecDimBlock>>>(d_r, d_r, d_rr_new, size);

        // beta_k = ||r_k+1||^2 / ||r_k||^2
        // beta = rr_new / rr;
        divide<<<1, 1>>>(d_rr_new, d_rr, d_beta);

        // update residual norm
        // rr = rr_new;
        memCopy<<<1, 1>>>(d_rr_new, d_rr, size);

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
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_p, d_temp, d_beta, size);
        vecVecAdd<<<vecDimGrid, vecDimBlock>>>(d_r, d_temp, d_p, size);
        // copy value of d_beta to dd_beta
        // cudaMemcpy(&dd_beta, d_beta, sizeof(double), cudaMemcpyDeviceToDevice);
        // cublasDaxpy(handle, SIZE, &dd_beta, d_p, 1, d_r, 1);
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
    // cudaDeviceReset();
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
}

// main function
void kernel_wrapper(double *matrix, double *rhs, double *sol, size_t size, int max_iters, double rel_error)
{
    printf("Calling kernel!\n");
    // solve_cuda(matrix, rhs, sol, size, max_iters, rel_error);
    solve_cublas(matrix, rhs, sol, size, max_iters, rel_error);
}
