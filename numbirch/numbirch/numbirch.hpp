/**
 * @file
 * 
 * NumBirch interface.
 */
#pragma once

#include <cstddef>

namespace numbirch {
/**
 * Initialize NumBirch. This should be called once at the start of the
 * program. It initializes, for example, thread-local variables necessary for
 * computations.
 */
void init();

/**
 * Terminate NumBirch.
 */
void term();

/**
 * Allocate memory.
 * 
 * @param bytes Number of bytes to allocate.
 * 
 * @return New allocation.
 */
void* malloc(const size_t size);

/**
 * Reallocate memory.
 * 
 * @param ptr Existing allocation.
 * @param size New size of allocation.
 * 
 * @return Resized allocation.
 */
void* realloc(void* ptr, const size_t size);

/**
 * Free allocation.
 * 
 * @param ptr Existing allocation.
 */
void free(void* ptr);

/**
 * Batch copy.
 * 
 * @param[out] dst Destination.
 * @param dpitch Stride between batches of `dst`, in bytes.
 * @param src Source.
 * @param spitch Stride between batches of `src`, in bytes.
 * @param width Width of each batch, in bytes.
 * @param height Number of batches.
 */
void memcpy(void* dst, const size_t dpitch, const void* src,
    const size_t spitch, const size_t width, const size_t height);

/**
 * Synchronize with the device. This waits for all operations to complete for
 * the current thread.
 */
void wait();

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
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
void neg(const int m, const int n, const double* A, const int ldA, double* B,
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
 * @param ldA Column stride of `A`.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
void sub(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC);

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
 * @param ldA Column stride of `A`.
 * @param b Matrix.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
void div(const int m, const int n, const double* A, const int ldA,
    const double b, double* C, const int ldC);

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
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
void mul(const int m, const int n, const double a, const double* B,
    const int ldB, double* C, const int ldC);

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
 * @param ldA Column stride of `A`.
 * 
 * @return Sum of elements of the matrix.
 */
double sum(const int m, const int n, const double* A, const int ldA);

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

/**
 * Outer product of matrix and lower-triangular Cholesky factor of another
 * matrix. Computes @f$C = AL^\top@f$, where @f$S = LL^\top@f$.
 * 
 * @param m Number of rows of `C`, and rows of `A`.
 * @param n Number of columns of `C` and rows of `S`.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param S Symmetric positive definite matrix.
 * @param ldS Column stride of `S`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
void cholouter(const int m, const int n, const double* A, const int ldA,
    const double* S, const int ldS, double* C, const int ldC);

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

}
