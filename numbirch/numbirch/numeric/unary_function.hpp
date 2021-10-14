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
 * @tparam T Value type.
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
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void acos(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Arc sine.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void asin(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Arc tangent.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void atan(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Round to smallest integer value not less than argument.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
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
 * @tparam T Value type.
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
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void cos(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Hyperbolic cosine.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void cosh(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Digamma function.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void digamma(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Exponential.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void exp(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Exponential minus one.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void expm1(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Round to largest integer value not greater than argument.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
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
 * @tparam T Value type.
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
 * @tparam T Value type.
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
 * @tparam T Value type.
 * 
 * @param n Number of rows and columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] b Logarithm of the absolute value of the determinant of `A`.
 */
template<class T>
void ldet(const int n, const T* A, const int ldA, T* b);

/**
 * Logarithm of the gamma function.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void lgamma(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Logarithm.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void log(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Logarithm of one plus argument.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void log1p(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Rectification. For element @f$(i,j)@f$, computes @f$B_{ij} = \max(A_{ij},
 * 0)@f$.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
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
 * @tparam T Value type.
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
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void sin(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Hyperbolic sine.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void sinh(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Square root.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void sqrt(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Sum of elements.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
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
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void tan(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Hyperbolic tangent.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] B Matrix.
 * @param ldB Column stride of `B`.
 */
template<class T, class U>
void tanh(const int m, const int n, const T* A, const int ldA, U* B,
    const int ldB);

/**
 * Matrix trace.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
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
 * @tparam T Value type.
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
