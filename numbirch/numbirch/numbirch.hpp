/**
 * @file
 * 
 * NumBirch C++ interface.
 */
#pragma once

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

/**
 * Matrix negation.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Row stride of `B`.
 */
void neg(const int m, const int n, const double* A, const int ldA,
    double* B, const int ldB);

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

/**
 * Matrix addition.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param B Matrix.
 * @param ldB Row stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Row stride of `C`.
 */
void add(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC);

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

/**
 * Matrix subtraction.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param B Matrix.
 * @param ldB Row stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Row stride of `C`.
 */
void sub(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC);

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

/**
 * Scalar-matrix multiplication.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param a Scalar.
 * @param B Matrix.
 * @param ldB Row stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Row stride of `C`.
 */
void mul(const int m, const int n, const double a, const double* B,
    const int ldB, double* C, const int ldC);

/**
 * Matrix-vector multiplication. Computes @f$y = Ax@f$.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param[out] y Vector.
 * @param incy Element stride of `y`.
 */
void mul(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy);

/**
 * Matrix-matrix multiplication. Computes @f$C = AB@f$.
 * 
 * @param m Number of rows of `A` and `C`.
 * @param n Number of columns of `B` and `C`.
 * @param k Number of columns of `A` and rows of `B`.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param B Matrix.
 * @param ldB Row stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Row stride of `C`.
 */
void mul(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC);

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

/**
 * Matrix-scalar division.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param b Matrix.
 * @param[out] C Matrix.
 * @param ldC Row stride of `C`.
 */
void div(const int m, const int n, const double* A, const int ldA,
    const double b, double* C, const int ldC);

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

/**
 * Matrix sum of elements.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * 
 * @return Sum of elements of the matrix.
 */
double sum(const int m, const int n, const double* A, const int ldA);

/**
 * Vector trace (sum of elements).
 * 
 * @param n Number of elements.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * 
 * @return Trace of the vector.
 */
double trace(const int n, const double* x, const int incx);

/**
 * Matrix trace (sum of elements on main diagonal).
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * 
 * @return Trace of the matrix.
 */
double trace(const int m, const int n, const double* A, const int ldA);

/**
 * Vector dot product. Computes @f$x^\top x@f$, resulting in a scalar.
 * 
 * @param n Number of elements.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * 
 * @return Dot product.
 */
double dot(const int n, const double* x, const int incx);

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

/**
 * Vector-matrix dot product. Computes @f$y = (x^\top A)^\top@f$, resulting in
 * a vector.
 * 
 * @param m Number of elements of `y` and columns of `A`.
 * @param n Number of elements of `x` and rows of `A`.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param A Matrix.
 * @param ldA Rows stride of `A`.
 * @param[out] y Vector.
 * @param incy Element stride of `y`.
 */
void dot(const int m, const int n, const double* x, const int incx,
    const double* A, const int ldA, double* y, const int incy);

/**
 * Vector inner product. Computes @f$y = x^\top x@f$, resulting in a vector of
 * length one.
 * 
 * @param n Number of elements.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param[out] y Vector.
 */
void inner(const int n, const double* x, const int incx, double* y);

/**
 * Matrix inner product. Computes @f$B = A^\top A@f$.
 * 
 * @param m Number of rows and columns of `B`, and columns of `A`.
 * @param n Number of rows of `A`.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Row stride of `B`.
 */
void inner(const int m, const int n, const double* A, const int ldA,
    double* B, const int ldB);

/**
 * Vector-vector inner product. Computes @f$z = x^\top y@f$, resulting in a
 * vector of length one.
 * 
 * @param n Number of elements.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param y Vector.
 * @param incy Element stride of `y`.
 * @param[out] z Vector.
 */
void inner(const int n, const double* x, const int incx, const double* y,
    const int incy, double* z);

/**
 * Matrix-vector inner product. Computes @f$y = A^\top x@f$.
 * 
 * @param m Number of elements of `y` and columns of `A`.
 * @param n Number of elements of `x` and rows of `A`.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param[out] y Vector.
 * @param incy Element stride of `y`.
 */
void inner(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy);

/**
 * Vector-matrix inner product. Computes @f$B = x^\top A@f$, resulting in a
 * matrix of one row.
 * 
 * @param n Number of columns of `A`.
 * @param k Number of elements of `x` and rows of `A`.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param[out] B Matrix.
 */
void inner(const int n, const int k, const double* x, const int incx,
    const double* A, const int ldA, double* B);

/**
 * Matrix-matrix inner product. Computes @f$C = A^\top B@f$.
 * 
 * @param m Number of rows of `C`, and columns of `A`.
 * @param n Number of columns of `C` and columns of `B`.
 * @param k Number of rows of `A` and rows of `B`.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param B Matrix.
 * @param ldB Row stride of `B`.
 * @param C Matrix.
 * @param ldC Row stride of `C`.
 */
void inner(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC);

/**
 * Vector outer product. Computes @f$A = xx^\top@f$.
 * 
 * @param n Number of rows and columns of `A`, and elements of `x`.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param[out] A Matrix.
 * @param ldA Row stride of `A`.
 */
void outer(const int n, const double* x, const int incx, double* A,
    const int ldA);

/**
 * Matrix outer product. Computes @f$B = AA^\top@f$.
 * 
 * @param n Number of rows and columns of `B`, and rows of `A`.
 * @param k Number of columns of `A`.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Row stride of `B`.
 */
void outer(const int n, const int k, const double* A, const int ldA,
    double* B, const int ldB);

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
 * @param ldA Row stride of `A`.
 */
void outer(const int m, const int n, const double* x, const int incx,
    const double* y, const int incy, double* A, const int ldA);

/**
 * Matrix-matrix outer product. Computes @f$C = AB^\top@f$.
 * 
 * @param m Number of rows of `C`, and rows of `A`.
 * @param n Number of columns of `C` and rows of `B`.
 * @param k Number of columns of `A` and columns of `B`.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param B Matrix.
 * @param ldB Row stride of `B`.
 * @param C Matrix.
 * @param ldC Row stride of `C`.
 */
void outer(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC);

/**
 * Matrix-vector solve. Solves for @f$x@f$ in @f$Ax = y@f$.
 * 
 * @param n Number of elements of `x`.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param[out] x Vector.
 * @param incx Element stride of `x`.
 * @param y Vector.
 * @param incy Element stride of `y`.
 */
void solve(const int n, const double* A, const int ldA, double* x,
    const int incx, const double* y, const int incy);

/**
 * Matrix-matrix solve. Solves for @f$X@f$ in @f$AX = Y@f$.
 * 
 * @param m Number of rows of `X`.
 * @param n Number of columns of `X`.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param[out] X Matrix.
 * @param ldX Row stride of `X`.
 * @param Y Matrix.
 * @param ldY Row stride of `Y`.
 */
void solve(const int m, const int n, const double* A, const int ldA,
    double* X, const int ldX, const double* Y, const int ldY);

/**
 * Matrix-vector solve, via the Cholesky factorization. Solves for @f$x@f$ in
 * @f$Sx = y@f$.
 * 
 * @param n Number of elements of `x`.
 * @param S Symmetric positive definite matrix.
 * @param ldS Row stride of `S`.
 * @param[out] x Vector.
 * @param incx Element stride of `x`.
 * @param y Vector.
 * @param incy Element stride of `y`.
 */
void cholsolve(const int n, const double* S, const int ldS, double* x,
    const int incx, const double* y, const int incy);

/**
 * Matrix-matrix solve, via the Cholesky factorization. Solves for @f$X@f$ in
 * @f$SX = Y@f$.
 * 
 * @param m Number of rows of `X`.
 * @param n Number of columns of `X`.
 * @param S Symmetric positive definite matrix.
 * @param ldS Row stride of `S`.
 * @param[out] X Matrix.
 * @param ldX Row stride of `X`.
 * @param Y Matrix.
 * @param ldY Row stride of `Y`.
 */
void cholsolve(const int m, const int n, const double* S, const int ldS,
    double* X, const int ldX, const double* Y, const int ldY);

/**
 * Inverse of a square matrix.
 * 
 * @param n Number of rows and columns.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Row stride of `B`.
 */
void inv(const int n, const double* A, const int ldA, double* B,
    const int ldB);

/**
 * Inverse of a square matrix, via the Cholesky factorization.
 * 
 * @param n Number of rows and columns.
 * @param S Symmetric positive definite matrix.
 * @param ldS Row stride of `S`.
 * @param[out] B Matrix.
 * @param ldB Row stride of `B`.
 */
void cholinv(const int n, const double* S, const int ldS, double* B,
    const int ldB);

/**
 * Logarithm of the absolute value of the determinant of a matrix.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * 
 * @return Logarithm of the absolute value of the determinant of `A`.
 */
double ldet(const int m, const int n, const double* A, const int ldA);

/**
 * Logarithm of the determinant of a matrix, via the Cholesky factorization.
 * 
 * @param n Number of rows and columns.
 * @param S Symmetric positive definite matrix.
 * @param ldS Row stride of `S`.
 * 
 * @return Logarithm of the determinant of `S`.
 * 
 * The determinant of a positive definite matrix is always positive.
 */
double lcholdet(const int n, const double* S, const int ldS);

/**
 * Vector transpose.
 * 
 * @param n Number of elements.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param[out] B Matrix.
 * @param ldB Row stride of `B`.
 */
void transpose(const int n, const double* x, const int incx, double* B,
    const int ldB);

/**
 * Matrix transpose.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Row stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Row stride of `B`.
 */
void transpose(const int m, const int n, const double* A, const int ldA,
    double* B, const int ldB);

}
