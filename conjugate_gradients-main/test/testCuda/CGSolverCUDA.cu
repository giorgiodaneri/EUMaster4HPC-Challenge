#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <chrono>

#include <cuda_runtime.h>

extern "C++"
{
#include "../../include/CGSolver.hpp"
#include "../../include/CGSolverCuda.hpp"
}

// vector vector multiply
#define BLOCK_SIZE 128
#define SIZE 10000 // TODO: handle this better, maybe pass it as a parameter to the functions

// __global__ void vecVecMult(double *a, double *b, float *c)
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

// dot product
__global__ void dotProduct(double *a, double *b, float *c)
{
    // __shared__ double temp[BLOCK_SIZE];
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // temp[threadIdx.x] = a[i] * b[i];
    // __syncthreads();
    // if (threadIdx.x == 0)
    // {
    //     float sum = 0;
    //     for (int j = 0; j < blockDim.x; j++)
    //     {
    //         sum += temp[j];
    //     }
    //     atomicAdd(c, sum);
    // }
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	float tmp = 0.0;
	if (i == 0) {
		for (int i = 0; i < SIZE; i++) {
			tmp += b[i] * a[i];
		}
		*c = tmp;
	}
}

// scalar vector multiply
__global__ void scalarVecMult(double *a, double *b, float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SIZE)
    {
        b[i] = a[i] * *c;
    }
}

// divide two scalars
__global__ void divide(float *a, float *b, float *c)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0)
        *c = *a / *b;
}

__global__ void memCopy(float *in, float *out) {
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < SIZE) {
		out[index] = in[index];
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

    // residual
    double *r = new double[size];
    // preconditioned residual
    double *p = new double[size];
    double *Ap = new double[size];
    int num_iters;
    // norm of the residual
    float* r_norm = (float *) malloc(sizeof(float));
    float* b_norm = (float *) malloc(sizeof(float));
    *r_norm = 1.0; 
    float rel_error = static_cast<float>(tolerance);

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

    // allocate memory for five floats, since cuda needs this type for atomicAdd, rather than double
    // (no overloading exists for variables of type double)
    float *d_alpha, *d_beta, *d_rr, *d_rr_new, *d_bb, *d_temp_scalar;
    cudaMalloc((void **)&d_alpha, sizeof(float));
    cudaMalloc((void **)&d_beta, sizeof(float));
    cudaMalloc((void **)&d_rr, sizeof(float));
    cudaMalloc((void **)&d_rr_new, sizeof(float));
    cudaMalloc((void **)&d_bb, sizeof(float));
    cudaMalloc((void **)&d_temp_scalar, sizeof(float));

    // initialize the residual by copying the rhs into it
    cudaMemcpy(d_r, d_b, size * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_p, d_b, size * sizeof(double), cudaMemcpyDeviceToDevice);
    
    // calculate the dot product of the rhs, which is equal to that
    // of the residual
    dotProduct<<<vecDimGrid, vecDimBlock>>>(d_b, d_b, d_bb);
    // copy value of d_bb to d_rr
    cudaMemcpy(d_rr, d_bb, sizeof(float), cudaMemcpyDeviceToDevice);
    // copy value of d_bb (norm of rhs) to host 
    cudaMemcpy(b_norm, d_bb, sizeof(float), cudaMemcpyDeviceToHost);
    
    // now start the CG solver iterations
    for(num_iters = 0; num_iters < maxIterations; num_iters++)
    {
        matVecMult<<<matDimGrid, matDimBlock>>>(d_A, d_p, d_Ap);
        // compute new alpha coefficient
        dotProduct<<<vecDimGrid, vecDimBlock>>>(d_p, d_Ap, d_temp_scalar);
        divide<<<1,1>>>(d_rr, d_temp_scalar, d_alpha);

        // compute new approximate of the solution at step k+1
        // x_k+1 = x_k + alpha_k * p_k
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_p, d_temp, d_alpha);
        vecVecAdd<<<vecDimGrid, vecDimBlock>>>(d_x, d_temp, d_x);

        // compute new residual at step k+1
        // r_k+1 = r_k - alpha_k * A * p_k
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_Ap, d_temp, d_alpha);
        vecVecSub<<<vecDimGrid, vecDimBlock>>>(d_r, d_temp, d_r);

        // update the 2-norm of the residual at step k+1
        dotProduct<<<vecDimGrid, vecDimBlock>>>(d_r, d_r, d_rr_new);

        // beta_k = ||r_k+1||^2 / ||r_k||^2
        divide<<<1,1>>>(d_rr_new, d_rr, d_beta);
        
        // update residual norm
        memCopy<<<1,1>>>(d_rr, d_rr_new);

        // stopping criterion ==> sqrt(||r||^2 / ||b||^2) < rel_error equivalent to 2-norm or euclidean norm
        // need to copy the norm of the residual back to the device
        // in order to evaluate it
        cudaMemcpy(r_norm, d_rr, sizeof(float), cudaMemcpyDeviceToHost);
        if (std::sqrt(*r_norm / *b_norm) < rel_error)
        {
            break;
        }

        // compute new direction at step k+1
        // p_k+1 = r_k+1 + beta_k * p_k
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_p, d_temp, d_beta);
        vecVecAdd<<<vecDimGrid, vecDimBlock>>>(d_r, d_temp, d_p);
    }
}

// main function
void kernel_wrapper(double* matrix, double* rhs, double *sol, size_t size, int max_iters, float rel_error)
{
    printf("Calling kernel!\n");
    solve_cuda(matrix, rhs, sol, size, max_iters, rel_error);
    // do work...
}
