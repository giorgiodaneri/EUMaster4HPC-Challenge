#ifndef CGSOLVERACC
#define CGSOLVERACC

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include "CGSolver.hpp"

class CGSolverACC : public CGSolver
{
public:
    CGSolverACC(double *A, double *b, double *x, size_t size, int maxIterations, double tolerance) : 
    CGSolver(A, b, x, size, maxIterations, tolerance) {}

    ~CGSolverACC() = default;

    void solve() override;
    double dot(const double *x, const double *y, size_t size) override;
    void axpby(double alpha, const double *x, double beta, double *y, size_t size) override;
    void precA(const double *A, const double *x, double *Ax, size_t size) override;
};
#endif