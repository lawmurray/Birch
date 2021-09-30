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
 * Negation.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void neg(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Rectification. For element @f$(i,j)@f$, computes @f$B_{ij} = \max(A_{ij},
 * 0)@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void rectify(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);


/**
 * Addition.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
void add(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC);

/**
 * Subtraction.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
void sub(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC);

/**
 * Linear combination of matrices.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
void combine(const int m, const int n, const T a, const T* A, const int ldA,
    const T b, const T* B, const int ldB, const T c, const T* C,
    const int ldC, const T d, const T* D, const int ldD, T* E,
    const int ldE);

/**
 * Hadamard (element-wise) matrix multiplication.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
void hadamard(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC);

/**
 * Scalar division.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param b Matrix.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
template<class T>
void div(const int m, const int n, const T* A, const int ldA, const T b, T* C,
    const int ldC);

/**
 * Scalar multiplication.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param a Scalar.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
template<class T>
void mul(const int m, const int n, const T a, const T* B, const int ldB, T* C,
    const int ldC);

/**
 * Matrix-vector multiplication. Computes @f$y = Ax@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
void mul(const int m, const int n, const T* A, const int ldA, const T* x,
    const int incx, T* y, const int incy);

/**
 * Matrix-matrix multiplication. Computes @f$C = AB@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
void mul(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC);

/**
 * Lower-triangular Cholesky factor of a matrix multiplied by a vector.
 * Computes @f$y = Lx@f$, where @f$S = LL^\top@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param n Number of rows and columns of `S`, and elements of `x` and `y`.
 * @param S Symmetric positive definite matrix.
 * @param ldS Column stride of `S`.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param[out] y Vector.
 * @param incy Element stride of `y`.
 */
template<class T>
void cholmul(const int n, const T* S, const int ldS, const T* x,
    const int incx, T* y, const int incy);

/**
 * Lower-triangular Cholesky factor of a matrix multiplied by a matrix.
 * Computes @f$C = LB@f$, where @f$S = LL^\top@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
void cholmul(const int m, const int n, const T* S, const int ldS, const T* B,
    const int ldB, T* C, const int ldC);

/**
 * Matrix sum of elements.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * 
 * @return Sum of elements of the matrix.
 */
template<class T>
T sum(const int m, const int n, const T* A, const int ldA);

/**
 * Vector-vector dot product. Computes @f$x^\top y@f$, resulting in a scalar.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param n Number of elements.
 * @param x Vector.
 * @param incx Element stride of `x`.
 * @param y Vector.
 * @param incy Element stride of `y`.
 * 
 * @return Dot product.
 */
template<class T>
T dot(const int n, const T* x, const int incx, const T* y, const int incy);

/**
 * Matrix-matrix Frobenius product. Computes @f$\langle A, B 
 * \rangle_\mathrm{F} = \mathrm{Tr}(A^\top B) = \sum_{ij} A_{ij} B_{ij}@f$,
 * resulting in a scalar.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
T frobenius(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB);

/**
 * Matrix-vector inner product. Computes @f$y = A^\top x@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
void inner(const int m, const int n, const T* A, const int ldA, const T* x,
    const int incx, T* y, const int incy);

/**
 * Matrix-matrix inner product. Computes @f$C = A^\top B@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
void inner(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC);

/**
 * Vector-vector outer product. Computes @f$A = xy^\top@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
void outer(const int m, const int n, const T* x, const int incx, const T* y,
    const int incy, T* A, const int ldA);

/**
 * Matrix-matrix outer product. Computes @f$C = AB^\top@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
void outer(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC);

/**
 * Outer product of matrix and lower-triangular Cholesky factor of another
 * matrix. Computes @f$C = AL^\top@f$, where @f$S = LL^\top@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
void cholouter(const int m, const int n, const T* A, const int ldA,
    const T* S, const int ldS, T* C, const int ldC);

/**
 * Matrix-vector solve. Solves for @f$x@f$ in @f$Ax = y@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param n Number of elements of `x`.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] x Vector.
 * @param incx Element stride of `x`.
 * @param y Vector.
 * @param incy Element stride of `y`.
 */
template<class T>
void solve(const int n, const T* A, const int ldA, T* x, const int incx,
    const T* y, const int incy);

/**
 * Matrix-matrix solve. Solves for @f$X@f$ in @f$AX = Y@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
void solve(const int m, const int n, const T* A, const int ldA, T* X,
    const int ldX, const T* Y, const int ldY);

/**
 * Matrix-vector solve, via the Cholesky factorization. Solves for @f$x@f$ in
 * @f$Sx = y@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param n Number of elements of `x`.
 * @param S Symmetric positive definite matrix.
 * @param ldS Column stride of `S`.
 * @param[out] x Vector.
 * @param incx Element stride of `x`.
 * @param y Vector.
 * @param incy Element stride of `y`.
 */
template<class T>
void cholsolve(const int n, const T* S, const int ldS, T* x, const int incx,
    const T* y, const int incy);

/**
 * Matrix-matrix solve, via the Cholesky factorization. Solves for @f$X@f$ in
 * @f$SX = Y@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
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
template<class T>
void cholsolve(const int m, const int n, const T* S, const int ldS, T* X,
    const int ldX, const T* Y, const int ldY);

/**
 * Inverse of a square matrix.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param n Number of rows and columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void inv(const int n, const T* A, const int ldA, T* B, const int ldB);

/**
 * Inverse of a symmetric positive definite square matrix, via the Cholesky
 * factorization.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param n Number of rows and columns.
 * @param S Symmetric positive definite matrix.
 * @param ldS Column stride of `S`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void cholinv(const int n, const T* S, const int ldS, T* B, const int ldB);

/**
 * Logarithm of the absolute value of the determinant of a square matrix.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param n Number of rows and columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * 
 * @return Logarithm of the absolute value of the determinant of `A`.
 */
template<class T>
T ldet(const int n, const T* A, const int ldA);

/**
 * Logarithm of the absolute value of the determinant of a symmetric positive
 * definite matrix, via the Cholesky factorization.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param n Number of rows and columns.
 * @param S Symmetric positive definite matrix.
 * @param ldS Column stride of `S`.
 * 
 * @return Logarithm of the determinant of `S`.
 * 
 * The determinant of a positive definite matrix is always positive.
 */
template<class T>
T lcholdet(const int n, const T* S, const int ldS);

/**
 * Scalar product and transpose. Computes @f$B = xA^\top@f$.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param m Number of rows of `B` and columns of `A`.
 * @param n Number of columns of `B` and rows of `A`.
 * @param x Scalar.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void transpose(const int m, const int n, const T x, const T* A, const int ldA,
    T* B, const int ldB);

/**
 * Matrix trace.
 * 
 * @ingroup cpp-low
 * 
 * @tparam T Value type (`double` or `float`).
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * 
 * @return Trace of the matrix.
 */
template<class T>
T trace(const int m, const int n, const T* A, const int ldA);

}
