#ifndef CGSOLVER
#define CGSOLVER

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <omp.h>

class CGSolver
{
public:
    CGSolver(double *A, double *b, double *x, size_t size, int maxIterations, double tolerance) : 
    A_(A), 
    b_(b), 
    x_(x), 
    size_(size), 
    max_iter_(maxIterations), 
    tol_(tolerance) {}

    ~CGSolver() = default;

    // void solve();
    // void setA(double *A) {A_ = A;}
    // void setB(double *b) {b_ = b;}
    // void setX(double *x) {x_ = x;}
    void setTolerance(double tolerance) {tol_ = tolerance;}
    void setMaxIter(int maxIterations) {max_iter_ = maxIterations;}
    double *getA() {return A_;}
    double *getB() {return b_;}
    double *getX() {return x_;}
    size_t getSize() {return size_;}
    int getMaxIter() {return max_iter_;}
    double getRelErr() {return tol_;}
    void solve();
    void solveOmp();
    void solveOpenACC();
    
protected:
    double dot(const double *x, const double *y, size_t size);
    void axpby(double alpha, const double *x, double beta, double *y, size_t size);
    void precA(const double *A, const double *x, double *Ax, size_t size);
    double dot_acc(const double *x, const double *y, size_t size);
    void axpby_acc(double alpha, const double *x, double beta, double *y, size_t size);
    void precA_acc(const double *A, const double *x, double *Ax, size_t size);


private:
    // system matrix A, square and spd
    double *A_;
    // right hand side
    double *b_;
    // solution vector
    double *x_;
    // size of the system
    size_t size_;
    // maximum number of iterations
    int max_iter_;
    // tolerance level used as stopping criterion
    double tol_;
};

#endif
