#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <chrono>
#include <openacc.h>
#include <mkl.h>
#include "CGSolver.hpp"
#include "CGSolverACC.hpp"

double CGSolverACC::dot(const double *x, const double *y, size_t size)
{
    double result = 0.0;
    // #pragma acc data copyin(x[0 : size], y[0 : size]) copy(result)
    //     {
    // #pragma acc parallel loop gang vector reduction(+ : result)
    for (size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    // } // end data region
    return result;
    // return cblas_ddot(size, x, 1, y, 1);
}

// #pragma acc routine seq
void CGSolverACC::axpby(double alpha, const double *x, double beta, double *y, size_t size)
{
    // #pragma acc data copyin(x[0 : size]) copy(y[0 : size])
    //     {
    // #pragma acc parallel loop gang vector present(x[0 : size])
    for (size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
    // } // end data region
}

// accelerated version of matrix vector multiply with offloading to GPU through OpenACC
void CGSolverACC::precA(const double *A, const double *x, double *Ax, size_t size)
{
#pragma acc data copyin(x[0 : size]) copyout(Ax[0 : size])
#pragma acc parallel loop gang vector independent present(A[0 : size * size])
    for (size_t i = 0; i < size; i++)
    {
        double y_val = 0.0;
        for (size_t j = 0; j < size; j++)
        {
            y_val += A[i * size + j] * x[j];
        }
        Ax[i] = y_val;
    }
}

void CGSolverACC::solve()
{
}

void CGSolverACC::solve_acc()
{
    // using namespace std::chrono;

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

    // // Get starting timepoint
    // auto start = high_resolution_clock::now();

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
        precA(A, p, Ap, size);

        // compute new alpha coefficient to guarantee optimal convergence rate
        alpha = rr / dot(p, Ap, size);

        // compute new approximate of the solution at step k+1
        // x_k+1 = x_k + alpha_k * p_k
        axpby(alpha, p, 1.0, x, size);

        // compute new residual at step k+1
        // r_k+1 = r_k - alpha_k * A * p_k
        axpby(-alpha, Ap, 1.0, r, size);

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

    // auto stop = high_resolution_clock::now();
    // // // Get duration. Substart timepoints to
    // // // get duration. To cast it to proper unit
    // // // use duration cast method
    // auto duration = duration_cast<milliseconds>(stop - start);
    // std::cout << "Time taken by function: "
    //           << duration.count() << " milliseconds" << std::endl;

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