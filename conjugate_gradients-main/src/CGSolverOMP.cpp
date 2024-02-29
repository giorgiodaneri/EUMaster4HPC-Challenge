#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <cblas.h>
#include "CGSolver.hpp"
#include "CGSolverOMP.hpp"

// dot product between two vectors
#pragma omp declare simd
double CGSolverOMP::dot(const double *x, const double *y, size_t size)
{
    double result = 0.0;
// also take into account omp simd directive
#pragma omp parallel for simd reduction(+ : result)
    for (size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}

#pragma omp declare simd
void CGSolverOMP::axpby(double alpha, const double *x, double beta, double *y, size_t size)
{
#pragma omp parallel for simd 
    for (size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

// TASK VERSION
// void CGSolverOMP::precA(const double *A, const double *x, double *Ax, size_t size)
// {
// // #pragma omp parallel for
// #pragma omp parallel
//     {

// #pragma omp single nowait
//         {

// #pragma omp taskloop nogroup
//             for (size_t i = 0; i < size; i++)
//             {
//                 double y_val = 0.0;
//                 // the following pragma is totally useless, why?
// #pragma omp simd reduction(+ : y_val)
//                 for (size_t j = 0; j < size; j++)
//                 {
//                     y_val += A[i * size + j] * x[j];
//                 }
//                 Ax[i] = y_val;
//             }
//         }
//     }
// }

// PARALLEL LOOPS VERSION
#pragma omp declare simd
void CGSolverOMP::precA(const double *A, const double *x, double *Ax, size_t size)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++)
    {
        double y_val = 0.0;
#pragma omp parallel for simd reduction(+ : y_val)
        for (size_t j = 0; j < size; j++)
        {
            y_val += A[i * size + j] * x[j];
        }
        Ax[i] = y_val;
    }
}

void CGSolverOMP::gemv(const double *A, const double *x, double *Ax, size_t size)
{
    // Set the number of threads for OpenBLAS
    // openblas_set_num_threads(omp_get_max_threads());

    // Matrix-vector multiplication using OpenBLAS's cblas_dgemv function
    cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1.0, A, size, x, 1, 0.0, Ax, 1);
}

void CGSolverOMP::solve()
{
    double *A = getA();
    double *b = getB();
    double *x = getX();
    size_t size = getSize();
    int max_iters = getMaxIter();
    double rel_error = getRelErr();

    double alpha, beta, bb, rr, rr_new;
    // residual
    double *r = new double[size];
    // preconditioned residual
    double *p = new double[size];
    double *Ap = new double[size];
    int num_iters;

#pragma omp parallel for simd
    for (size_t i = 0; i < size; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    // needed for stopping criterion
    bb = dot(b, b, size);
    rr = bb;

    for (num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        // brainy way to compute A * p, need it for residual update and computation of alpha
        // writes result directly in Ap
        // is this some kind of obfuscation to make the code less readable???
        // gemv(1.0, A, p, 0.0, Ap, size, size);
        precA(A, p, Ap, size);
        // compute new alpha coefficient to guarantee optimal convergence rate
        alpha = rr / dot(p, Ap, size);

#pragma omp parallel sections
        {
#pragma omp section
            {
                // compute new approximate of the solution at step k+1
                // x_k+1 = x_k + alpha_k * p_k
                axpby(alpha, p, 1.0, x, size);
            }
#pragma omp section
            {
                // compute new residual at step k+1
                // r_k+1 = r_k - alpha_k * A * p_k
                axpby(-alpha, Ap, 1.0, r, size);
            }
        }

        // update the 2-norm of the residual at step k+1
        rr_new = dot(r, r, size);
        // compute beta coefficient
        // beta_k = ||r_k+1||^2 / ||r_k||^2
        // stopping criterion ==> sqrt(||r||^2 / ||b||^2) < rel_error equivalent to 2-norm or euclidean norm
        beta = rr_new / rr;
        rr = rr_new;
        if (std::sqrt(rr / bb) < rel_error)
        {
            break;
        }

        // compute new direction at step k+1
        // p_k+1 = r_k+1 + beta_k * p_k
        axpby(1.0, r, beta, p, size);
    }

    delete[] r;
    delete[] p;
    delete[] Ap;

    if (num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
}

void CGSolverOMP::solveBLAS()
{
    double *A = getA();
    double *b = getB();
    double *x = getX();
    size_t size = getSize();
    int max_iters = getMaxIter();
    double rel_error = getRelErr();

    double alpha, beta, bb, rr, rr_new;
    // residual
    double *r = new double[size];
    // preconditioned residual
    double *p = new double[size];
    double *Ap = new double[size];
    int num_iters;

#pragma omp parallel for simd
    for (size_t i = 0; i < size; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    // needed for stopping criterion
    bb = dot(b, b, size);
    rr = bb;

    for (num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        gemv(A, p, Ap, size);
        // compute new alpha coefficient to guarantee optimal convergence rate
        alpha = rr / dot(p, Ap, size);

#pragma omp parallel sections
        {
#pragma omp section
            {
                // compute new approximate of the solution at step k+1
                // x_k+1 = x_k + alpha_k * p_k
                axpby(alpha, p, 1.0, x, size);
            }
#pragma omp section
            {
                // compute new residual at step k+1
                // r_k+1 = r_k - alpha_k * A * p_k
                axpby(-alpha, Ap, 1.0, r, size);
            }
        }

        // update the 2-norm of the residual at step k+1
        rr_new = dot(r, r, size);
        // compute beta coefficient
        // beta_k = ||r_k+1||^2 / ||r_k||^2
        // stopping criterion ==> sqrt(||r||^2 / ||b||^2) < rel_error equivalent to 2-norm or euclidean norm
        beta = rr_new / rr;
        rr = rr_new;
        if (std::sqrt(rr / bb) < rel_error)
        {
            break;
        }

        // compute new direction at step k+1
        // p_k+1 = r_k+1 + beta_k * p_k
        axpby(1.0, r, beta, p, size);
    }

    delete[] r;
    delete[] p;
    delete[] Ap;

    if (num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
}