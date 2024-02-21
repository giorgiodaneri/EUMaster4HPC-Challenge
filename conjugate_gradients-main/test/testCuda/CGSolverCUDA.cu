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
    __shared__ double temp[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    temp[threadIdx.x] = a[i] * b[i];
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float sum = 0;
        for (int j = 0; j < blockDim.x; j++)
        {
            sum += temp[j];
        }
        atomicAdd(c, sum);
    }
    // unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // float tmp = 0.0;
    // if (i == 0)
    // {
    //     for (int i = 0; i < SIZE; i++)
    //     {
    //         tmp += b[i] * a[i];
    //     }
    //     *c = tmp;
    // }
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

__global__ void memCopy(float *in, float *out)
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

    // residual
    double *r = new double[size];
    // preconditioned residual
    double *p = new double[size];
    double *Ap = new double[size];
    int num_iters;
    // norm of the residual
    float *r_norm = (float *)malloc(sizeof(float));
    float *b_norm = (float *)malloc(sizeof(float));
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
    cudaDeviceSynchronize();
    // copy value of d_bb to d_rr
    cudaMemcpy(d_rr, d_bb, sizeof(float), cudaMemcpyDeviceToDevice);
    // copy value of d_bb (norm of rhs) to host
    cudaMemcpy(b_norm, d_bb, sizeof(float), cudaMemcpyDeviceToHost);

    // now start the CG solver iterations
    for (num_iters = 0; num_iters < maxIterations; num_iters++)
    {
        // precA(A, p, Ap, size);
        matVecMult<<<matDimGrid, matDimBlock>>>(d_A, d_p, d_Ap);

        cudaDeviceSynchronize();

        // compute new alpha coefficient
        // alpha = rr / dot(p, Ap, size);
        dotProduct<<<vecDimGrid, vecDimBlock>>>(d_p, d_Ap, d_temp_scalar);
        cudaDeviceSynchronize();
        divide<<<1, 1>>>(d_rr, d_temp_scalar, d_alpha);

        cudaDeviceSynchronize();

        // compute new approximate of the solution at step k+1
        // x_k+1 = x_k + alpha_k * p_k
        // axpby(alpha, p, 1.0, x, size);
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_p, d_temp, d_alpha);
        cudaDeviceSynchronize();
        vecVecAdd<<<vecDimGrid, vecDimBlock>>>(d_x, d_temp, d_x);

        // compute new residual at step k+1
        // r_k+1 = r_k - alpha_k * A * p_k
        // axpby(-alpha, Ap, 1.0, r, size);
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_Ap, d_temp, d_alpha);
        cudaDeviceSynchronize();
        vecVecSub<<<vecDimGrid, vecDimBlock>>>(d_r, d_temp, d_r);

        cudaDeviceSynchronize();

        // update the 2-norm of the residual at step k+1
        // rr_new = dot(r, r, size);
        dotProduct<<<vecDimGrid, vecDimBlock>>>(d_r, d_r, d_rr_new);

        cudaDeviceSynchronize();

        // beta_k = ||r_k+1||^2 / ||r_k||^2
        // beta = rr_new / rr;
        divide<<<1, 1>>>(d_rr_new, d_rr, d_beta);

        // update residual norm
        // rr = rr_new;
        memCopy<<<1, 1>>>(d_rr, d_rr_new);

        cudaDeviceSynchronize();

        // stopping criterion ==> sqrt(||r||^2 / ||b||^2) < rel_error equivalent to 2-norm or euclidean norm
        // need to copy the norm of the residual back to the device
        // in order to evaluate it
        cudaMemcpy(r_norm, d_rr, sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if (std::sqrt(*r_norm / *b_norm) < rel_error)
        {
            break;
        }

        // compute new direction at step k+1
        // p_k+1 = r_k+1 + beta_k * p_k
        // axpby(1.0, r, beta, p, size);
        scalarVecMult<<<vecDimGrid, vecDimBlock>>>(d_p, d_temp, d_beta);
        cudaDeviceSynchronize();
        vecVecAdd<<<vecDimGrid, vecDimBlock>>>(d_r, d_temp, d_p);
        cudaDeviceSynchronize();
    }

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

    // free host memory
    free(r_norm);
    free(b_norm);
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
    double alpha, beta, bb, rr, rr_new;
    // residual
    double *r = new double[size];
    // preconditioned residual
    double *p = new double[size];
    double *Ap = new double[size];
    int num_iters;
    float *temp_scalar = (float *)malloc(sizeof(float));

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
    double *d_A, *d_p, *d_Ap;
    float *d_temp_scalar;
    cudaMalloc((void **)&d_temp_scalar, sizeof(float));
    cudaMalloc((void **)&d_A, size * size * sizeof(double));
    cudaMalloc((void **)&d_p, size * sizeof(double));
    cudaMalloc((void **)&d_Ap, size * sizeof(double));

    // copy data to device
    cudaMemcpy(d_A, A, size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ap, Ap, size * sizeof(double), cudaMemcpyHostToDevice);

    // now start the CG solver iterations
    for (num_iters = 0; num_iters < maxIterations; num_iters++)
    {
        // brainy way to compute A * p, need it for residual update and computation of alpha
        // writes result directly in Ap
        // is this some kind of obfuscation to make the code less readable???
        // gemv(1.0, A, p, 0.0, Ap, size, size);
        // precA(A, p, Ap, size);
        cudaMemcpy(d_p, p, size * sizeof(double), cudaMemcpyHostToDevice);

        matVecMult<<<matDimGrid, matDimBlock>>>(d_A, d_p, d_Ap);

        cudaMemcpy(Ap, d_Ap, size * sizeof(double), cudaMemcpyDeviceToHost);

        // compute new alpha coefficient to guarantee optimal convergence rate
        // dotProduct<<<vecDimGrid, vecDimBlock>>>(d_p, d_Ap, d_temp_scalar);
        // cudaDeviceSynchronize();
        // cudaMemcpy(temp_scalar, d_temp_scalar, sizeof(double), cudaMemcpyDeviceToHost);
        // alpha = rr / *temp_scalar;

        alpha = rr / dot(p, Ap, size);

        // compute new approximate of the solution at step k+1
        // x_k+1 = x_k + alpha_k * p_k
        axpby(alpha, p, 1.0, x, size);

        // compute new residual at step k+1
        // r_k+1 = r_k - alpha_k * A * p_k
        axpby(-alpha, Ap, 1.0, r, size);

        // compute the 2-norm of the residual at step k+1
        rr_new = dot(r, r, size);
        // compute beta coefficient
        // beta_k = ||r_k+1||^2 / ||r_k||^2
        beta = rr_new / rr;
        // update residual norm
        rr = rr_new;
        // stopping criterion ==> sqrt(||r||^2 / ||b||^2) < rel_error equivalent to 2-norm or euclidean norm
        if (std::sqrt(rr / bb) < tolerance)
        {
            break;
        }

        // compute new direction at step k+1
        // p_k+1 = r_k+1 + beta_k * p_k
        axpby(1.0, r, beta, p, size);
    }

    printf("relative error is: %f \n", rr / bb);
    printf("number of iterations: %d \n", num_iters);
    // free memory allocated on the device
    cudaFree(d_A);
    cudaFree(d_Ap);
    cudaFree(d_p);
    cudaFree(d_temp_scalar);

    delete[] r;
    delete[] p;
    delete[] Ap;
    free(temp_scalar);
}

// main function
void kernel_wrapper(double *matrix, double *rhs, double *sol, size_t size, int max_iters, float rel_error)
{
    printf("Calling kernel!\n");
    solve_try(matrix, rhs, sol, size, max_iters, rel_error);
}