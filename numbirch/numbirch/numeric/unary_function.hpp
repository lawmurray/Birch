/**
 * @file
 */
#pragma once

namespace numbirch {
/**
 * Absolute value.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Arithmetic type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void abs(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Arc cosine.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void acos(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Arc sine.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void asin(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Arc tangent.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void atan(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Round to smallest integer value not less than argument.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void ceil(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Inverse of a symmetric positive definite square matrix, via the Cholesky
 * factorization.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
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
 * Cosine.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void cos(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Hyperbolic cosine.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void cosh(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Count of non-zero elements.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Arithmetic type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] b Count of non-zero elements in the matrix.
 */
template<class T>
void count(const int m, const int n, const T* A, const int ldA, int* b);

/**
 * Construct diagonal matrix. Diagonal elements are assigned to a given scalar
 * value, while all off-diagonal elements are assigned zero.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param a Scalar to assign to diagonal.
 * @param n Number of rows and columns of `B`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void diagonal(const T* a, const int n, T* B, const int ldB);

/**
 * Digamma function.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void digamma(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Exponential.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void exp(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Exponential minus one.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void expm1(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Round to largest integer value not greater than argument.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void floor(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Inverse of a square matrix.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
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
 * Logarithm of the determinant of a symmetric positive definite matrix, via
 * the Cholesky factorization. The determinant of a positive definite matrix
 * is always positive.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param n Number of rows and columns.
 * @param S Symmetric positive definite matrix.
 * @param ldS Column stride of `S`.
 * @param[out] b Logarithm of the determinant of `S`.
 */
template<class T>
void lcholdet(const int n, const T* S, const int ldS, T* b);

/**
 * Logarithm of the absolute value of the determinant of a square matrix.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param n Number of rows and columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] b Logarithm of the absolute value of the determinant of `A`.
 */
template<class T>
void ldet(const int n, const T* A, const int ldA, T* b);

/**
 * Logarithm of the factorial function.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void lfact(const int m, const int n, const int* A, const int ldA, T* B,
    const int ldB);

/**
 * Gradient of lfact().
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param G Matrix.
 * @param ldG Column stride of `G`.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] GA Matrix.
 * @param ldGA Column stride of `GA`.
 */
template<class T>
void lfact_grad(const int m, const int n, const T* G, const int ldG,
    const int* A, const int ldA, T* GA, const int ldGA);

/**
 * Logarithm of the gamma function.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void lgamma(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Logarithm.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void log(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Logarithm of one plus argument.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void log1p(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Reciprocal. For element @f$(i,j)@f$, computes @f$B_{ij} = 1/A_{ij}@f$. The
 * division is as for the type `T`; this will always return zero for an
 * integer type.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void rcp(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Rectification. For element @f$(i,j)@f$, computes @f$B_{ij} = \max(A_{ij},
 * 0)@f$.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
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
 * Round to nearest integer value.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void round(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Sine.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void sin(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Construct single-entry vector. One of the elements of the vector is one,
 * all others are zero.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param i Index of single entry (1-based).
 * @param n Length of vector.
 * @param[out] x Vector.
 * @param incx Element stride of `x`.
 */
template<class T>
void single(const int* i, const int n, T* x, const int incx);

/**
 * Hyperbolic sine.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void sinh(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Square root.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void sqrt(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Sum of elements.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Arithmetic type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] b Sum of elements of the matrix.
 */
template<class T>
void sum(const int m, const int n, const T* A, const int ldA, T* b);

/**
 * Tangent.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void tan(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Hyperbolic tangent.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void tanh(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

/**
 * Matrix trace.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] b Trace of the matrix.
 */
template<class T>
void trace(const int m, const int n, const T* A, const int ldA, T* b);

/**
 * Matrix transpose. Computes @f$B = A^\top@f$.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows of `B` and columns of `A`.
 * @param n Number of columns of `B` and rows of `A`.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T>
void transpose(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB);

}
