/**
 * @file
 * 
 * NumBirch interface.
 */
#pragma once

#include "numbirch/array/Array.hpp"
#include "numbirch/memory.hpp"

namespace numbirch {
/**
 * Vector negation.
 * 
 * @param n Number of elements.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param[out] y Vector.
 * @param incy Element stride of `y`.
 */
void neg(const int n, const double* x, const int incx, double* y,
    const int incy);
void neg(const int n, const float* x, const int incx, float* y,
    const int incy);

/**
 * Matrix negation.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
void neg(const int m, const int n, const double* A, const int ldA, double* B,
    const int ldB);
void neg(const int m, const int n, const float* A, const int ldA, float* B,
    const int ldB);

/**
 * Vector addition.
 * 
 * @param n Number of elements.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param y Vector.
 * @param incy Element stride of `y`.
 * @param[out] z Vector.
 * @param incz Element stride of `z`.
 */
void add(const int n, const double* x, const int incx, const double* y,
    const int incy, double* z, const int incz);
void add(const int n, const float* x, const int incx, const float* y,
    const int incy, float* z, const int incz);

/**
 * Matrix addition.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
void add(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC);
void add(const int m, const int n, const float* A, const int ldA,
    const float* B, const int ldB, float* C, const int ldC);

/**
 * Vector subtraction.
 * 
 * @param n Number of elements.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param y Vector.
 * @param incy Element stride of `y`.
 * @param[out] z Vector.
 * @param incz Element stride of `z`.
 */
void sub(const int n, const double* x, const int incx, const double* y,
    const int incy, double* z, const int incz);
void sub(const int n, const float* x, const int incx, const float* y,
    const int incy, float* z, const int incz);

/**
 * Matrix subtraction.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
void sub(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC);
void sub(const int m, const int n, const float* A, const int ldA,
    const float* B, const int ldB, float* C, const int ldC);

/**
 * Linear combination of matrices.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param a Coefficient on `A`.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param b Coefficient on `B`.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param c Coefficient on `C`.
 * @param C Matrix.
 * @param ldC Column stride of `C`.
 * @param d Coefficient on `D`.
 * @param D Matrix.
 * @param ldD Column stride of `D`.
 * @param[out] E Matrix.
 * @param ldE Column stride of `E`.
 */
void combine(const int m, const int n, const double a, const double* A,
    const int ldA, const double b, const double* B, const int ldB,
    const double c, const double* C, const int ldC, const double d,
    const double* D, const int ldD, double* E, const int ldE);
void combine(const int m, const int n, const float a, const float* A,
    const int ldA, const float b, const float* B, const int ldB,
    const float c, const float* C, const int ldC, const float d,
    const float* D, const int ldD, float* E, const int ldE);

/**
 * Hadamard (element-wise) vector multiplication.
 * 
 * @param n Number of elements.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param y Vector.
 * @param incy Element stride of `y`.
 * @param[out] z Vector.
 * @param incz Element stride of `z`.
 */
void hadamard(const int n, const double* x, const int incx, const double* y,
    const int incy, double* z, const int incz);
void hadamard(const int n, const float* x, const int incx, const float* y,
    const int incy, float* z, const int incz);

/**
 * Hadamard (element-wise) matrix multiplication.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
void hadamard(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC);
void hadamard(const int m, const int n, const float* A, const int ldA,
    const float* B, const int ldB, float* C, const int ldC);

/**
 * Vector-scalar division.
 * 
 * @param n Number of elements.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param y Scalar.
 * @param[out] z Vector.
 * @param incz Element stride of `z`.
 */
void div(const int n, const double* x, const int incx, const double y,
    double* z, const int incz);
void div(const int n, const float* x, const int incx, const float y,
    float* z, const int incz);

/**
 * Matrix-scalar division.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param b Matrix.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
void div(const int m, const int n, const double* A, const int ldA,
    const double b, double* C, const int ldC);
void div(const int m, const int n, const float* A, const int ldA,
    const float b, float* C, const int ldC);

/**
 * Scalar-vector multiplication.
 * 
 * @param n Number of elements.
 * @param x Scalar.
 * @param y Vector.
 * @param incy Element stride of `y`.
 * @param[out] z Vector.
 * @param incz Element stride of `z`.
 */
void mul(const int n, const double x, const double* y, const int incy,
    double* z, const int incz);
void mul(const int n, const float x, const float* y, const int incy,
    float* z, const int incz);

/**
 * Scalar-matrix multiplication.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param a Scalar.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
void mul(const int m, const int n, const double a, const double* B,
    const int ldB, double* C, const int ldC);
void mul(const int m, const int n, const float a, const float* B,
    const int ldB, float* C, const int ldC);

/**
 * Matrix-vector multiplication. Computes @f$y = Ax@f$.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param[out] y Vector.
 * @param incy Element stride of `y`.
 */
void mul(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy);
void mul(const int m, const int n, const float* A, const int ldA,
    const float* x, const int incx, float* y, const int incy);

/**
 * Matrix-matrix multiplication. Computes @f$C = AB@f$.
 * 
 * @param m Number of rows of `A` and `C`.
 * @param n Number of columns of `B` and `C`.
 * @param k Number of columns of `A` and rows of `B`.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
void mul(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC);
void mul(const int m, const int n, const int k, const float* A,
    const int ldA, const float* B, const int ldB, float* C, const int ldC);

/**
 * Lower-triangular Cholesky factor of a matrix multiplied by a vector.
 * Computes @f$y = Lx@f$, where @f$S = LL^\top@f$.
 * 
 * @param n Number of rows and columns of `S`, and elements of `x` and `y`.
 * @param S Symmetric positive definite matrix.
 * @param ldS Column stride of `S`.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param[out] y Vector.
 * @param incy Element stride of `y`.
 */
void cholmul(const int n, const double* S, const int ldS, const double* x,
    const int incx, double* y, const int incy);
void cholmul(const int n, const float* S, const int ldS, const float* x,
    const int incx, float* y, const int incy);

/**
 * Lower-triangular Cholesky factor of a matrix multiplied by a matrix.
 * Computes @f$C = LB@f$, where @f$S = LL^\top@f$.
 * 
 * @param m Number of rows of `S` and `C`.
 * @param n Number of columns of `B` and `C`.
 * @param S Symmetric positive definite matrix.
 * @param ldS Column stride of `S`.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
void cholmul(const int m, const int n, const double* S, const int ldS,
    const double* B, const int ldB, double* C, const int ldC);
void cholmul(const int m, const int n, const float* S, const int ldS,
    const float* B, const int ldB, float* C, const int ldC);

/**
 * Vector sum of elements.
 * 
 * @param n Number of elements.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * 
 * @return Sum of elements of the vector.
 */
double sum(const int n, const double* x, const int incx);
float sum(const int n, const float* x, const int incx);

/**
 * Matrix sum of elements.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * 
 * @return Sum of elements of the matrix.
 */
double sum(const int m, const int n, const double* A, const int ldA);
float sum(const int m, const int n, const float* A, const int ldA);

/**
 * Vector-vector dot product. Computes @f$x^\top y@f$, resulting in a scalar.
 * 
 * @param n Number of elements.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param y Vector.
 * @param incy Element stride of `y`.
 * 
 * @return Dot product.
 */
double dot(const int n, const double* x, const int incx, const double* y,
    const int incy);
float dot(const int n, const float* x, const int incx, const float* y,
    const int incy);

/**
 * Matrix-matrix Frobenius product. Computes @f$\langle A, B 
 * \rangle_\mathrm{F} = \mathrm{Tr}(A^\top B) = \sum_{ij} A_{ij} B_{ij}@f$,
 * resulting in a scalar.
 * 
 * @param m Number of rows of `A` and `B`.
 * @param n Number of columns of `A` and `B`.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * 
 * @return Frobenius product.
 */
double frobenius(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB);
float frobenius(const int m, const int n, const float* A, const int ldA,
    const float* B, const int ldB);

/**
 * Matrix-vector inner product. Computes @f$y = A^\top x@f$.
 * 
 * @param m Number of elements of `y` and columns of `A`.
 * @param n Number of elements of `x` and rows of `A`.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param[out] y Vector.
 * @param incy Element stride of `y`.
 */
void inner(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy);
void inner(const int m, const int n, const float* A, const int ldA,
    const float* x, const int incx, float* y, const int incy);

/**
 * Matrix-matrix inner product. Computes @f$C = A^\top B@f$.
 * 
 * @param m Number of rows of `C`, and columns of `A`.
 * @param n Number of columns of `C` and columns of `B`.
 * @param k Number of rows of `A` and rows of `B`.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param C Matrix.
 * @param ldC Column stride of `C`.
 */
void inner(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC);
void inner(const int m, const int n, const int k, const float* A,
    const int ldA, const float* B, const int ldB, float* C, const int ldC);

/**
 * Vector-vector outer product. Computes @f$A = xy^\top@f$.
 * 
 * @param m Number of rows of `A` and elements of `x`.
 * @param n Number of columns of `A` and elements of `y`.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param y Vector.
 * @param incy Element stride of `y`.
 * @param[out] A Matrix.
 * @param ldA Column stride of `A`.
 */
void outer(const int m, const int n, const double* x, const int incx,
    const double* y, const int incy, double* A, const int ldA);
void outer(const int m, const int n, const float* x, const int incx,
    const float* y, const int incy, float* A, const int ldA);

/**
 * Matrix-matrix outer product. Computes @f$C = AB^\top@f$.
 * 
 * @param m Number of rows of `C`, and rows of `A`.
 * @param n Number of columns of `C` and rows of `B`.
 * @param k Number of columns of `A` and columns of `B`.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param C Matrix.
 * @param ldC Column stride of `C`.
 */
void outer(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC);
void outer(const int m, const int n, const int k, const float* A,
    const int ldA, const float* B, const int ldB, float* C, const int ldC);

/**
 * Outer product of matrix and lower-triangular Cholesky factor of another
 * matrix. Computes @f$C = AL^\top@f$, where @f$S = LL^\top@f$.
 * 
 * @param m Number of rows of `C`, and rows of `A`.
 * @param n Number of columns of `C`, columns of `A`, and rows and columns of
 * `S`.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param S Symmetric positive definite matrix.
 * @param ldS Column stride of `S`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
void cholouter(const int m, const int n, const double* A, const int ldA,
    const double* S, const int ldS, double* C, const int ldC);
void cholouter(const int m, const int n, const float* A, const int ldA,
    const float* S, const int ldS, float* C, const int ldC);

/**
 * Matrix-vector solve. Solves for @f$x@f$ in @f$Ax = y@f$.
 * 
 * @param n Number of elements of `x`.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] x Vector.
 * @param incx Element stride of `x`.
 * @param y Vector.
 * @param incy Element stride of `y`.
 */
void solve(const int n, const double* A, const int ldA, double* x,
    const int incx, const double* y, const int incy);
void solve(const int n, const float* A, const int ldA, float* x,
    const int incx, const float* y, const int incy);

/**
 * Matrix-matrix solve. Solves for @f$X@f$ in @f$AX = Y@f$.
 * 
 * @param m Number of rows of `X`.
 * @param n Number of columns of `X`.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] X Matrix.
 * @param ldX Column stride of `X`.
 * @param Y Matrix.
 * @param ldY Column stride of `Y`.
 */
void solve(const int m, const int n, const double* A, const int ldA,
    double* X, const int ldX, const double* Y, const int ldY);
void solve(const int m, const int n, const float* A, const int ldA,
    float* X, const int ldX, const float* Y, const int ldY);

/**
 * Matrix-vector solve, via the Cholesky factorization. Solves for @f$x@f$ in
 * @f$Sx = y@f$.
 * 
 * @param n Number of elements of `x`.
 * @param S Symmetric positive definite matrix.
 * @param ldS Column stride of `S`.
 * @param[out] x Vector.
 * @param incx Element stride of `x`.
 * @param y Vector.
 * @param incy Element stride of `y`.
 */
void cholsolve(const int n, const double* S, const int ldS, double* x,
    const int incx, const double* y, const int incy);
void cholsolve(const int n, const float* S, const int ldS, float* x,
    const int incx, const float* y, const int incy);

/**
 * Matrix-matrix solve, via the Cholesky factorization. Solves for @f$X@f$ in
 * @f$SX = Y@f$.
 * 
 * @param m Number of rows of `X`.
 * @param n Number of columns of `X`.
 * @param S Symmetric positive definite matrix.
 * @param ldS Column stride of `S`.
 * @param[out] X Matrix.
 * @param ldX Column stride of `X`.
 * @param Y Matrix.
 * @param ldY Column stride of `Y`.
 */
void cholsolve(const int m, const int n, const double* S, const int ldS,
    double* X, const int ldX, const double* Y, const int ldY);
void cholsolve(const int m, const int n, const float* S, const int ldS,
    float* X, const int ldX, const float* Y, const int ldY);

/**
 * Inverse of a square matrix.
 * 
 * @param n Number of rows and columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
void inv(const int n, const double* A, const int ldA, double* B,
    const int ldB);
void inv(const int n, const float* A, const int ldA, float* B,
    const int ldB);

/**
 * Inverse of a square matrix, via the Cholesky factorization.
 * 
 * @param n Number of rows and columns.
 * @param S Symmetric positive definite matrix.
 * @param ldS Column stride of `S`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
void cholinv(const int n, const double* S, const int ldS, double* B,
    const int ldB);
void cholinv(const int n, const float* S, const int ldS, float* B,
    const int ldB);

/**
 * Logarithm of the absolute value of the determinant of a square matrix.
 * 
 * @param n Number of rows and columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * 
 * @return Logarithm of the absolute value of the determinant of `A`.
 */
double ldet(const int n, const double* A, const int ldA);
float ldet(const int n, const float* A, const int ldA);

/**
 * Logarithm of the determinant of a matrix, via the Cholesky factorization.
 * 
 * @param n Number of rows and columns.
 * @param S Symmetric positive definite matrix.
 * @param ldS Column stride of `S`.
 * 
 * @return Logarithm of the determinant of `S`.
 * 
 * The determinant of a positive definite matrix is always positive.
 */
double lcholdet(const int n, const double* S, const int ldS);
float lcholdet(const int n, const float* S, const int ldS);

/**
 * Scalar-matrix product and transpose. Computes @f$B = xA^\top@f$.
 * 
 * @param m Number of rows of `B` and columns of `A`.
 * @param n Number of columns of `B` and rows of `A`.
 * @param x Scalar.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
void transpose(const int m, const int n, const double x, const double* A,
    const int ldA, double* B, const int ldB);
void transpose(const int m, const int n, const float x, const float* A,
    const int ldA, float* B, const int ldB);

/**
 * Matrix trace.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * 
 * @return Trace of the matrix.
 */
double trace(const int m, const int n, const double* A, const int ldA);
float trace(const int m, const int n, const float* A, const int ldA);

}
