/**
 * @file
 */
#pragma once

namespace numbirch {
/**
 * Inverse of a symmetric positive definite square matrix, via the Cholesky
 * factorization.
 * 
 * @ingroup cpp-raw
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
 * Inverse of a square matrix.
 * 
 * @ingroup cpp-raw
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
 * Logarithm of the determinant of a symmetric positive definite matrix, via
 * the Cholesky factorization. The determinant of a positive definite matrix
 * is always positive.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type (`double` or `float`).
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
 * @tparam T Value type (`double` or `float`).
 * 
 * @param n Number of rows and columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param[out] b Logarithm of the absolute value of the determinant of `A`.
 */
template<class T>
void ldet(const int n, const T* A, const int ldA, T* b);

/**
 * Rectification. For element @f$(i,j)@f$, computes @f$B_{ij} = \max(A_{ij},
 * 0)@f$.
 * 
 * @ingroup cpp-raw
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
 * Sum of elements.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type (`double` or `float`).
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
 * Matrix trace.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type (`double` or `float`).
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
 * @tparam T Value type (`double` or `float`).
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
