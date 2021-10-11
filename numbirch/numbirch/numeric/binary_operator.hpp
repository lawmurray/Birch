/**
 * @file
 */
#pragma once

namespace numbirch {
/**
 * Addition.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * @tparam V Value type.
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
template<class T, class U, class V>
void add(const int m, const int n, const T* A, const int ldA, const U* B,
    const int ldB, V* C, const int ldC);

/**
 * Scalar division.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * @tparam V Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param b Scalar.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
template<class T, class U, class V>
void div(const int m, const int n, const T* A, const int ldA, const U* b,
    V* C, const int ldC);

/**
 * Equal to comparison.
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
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
template<class T, class U>
void equal(const int m, const int n, const T* A, const int ldA, const U* B,
    const int ldB, bool* C, const int ldC);

/**
 * Greater than comparison.
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
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
template<class T, class U>
void greater(const int m, const int n, const T* A, const int ldA, const U* B,
    const int ldB, bool* C, const int ldC);

/**
 * Greater than or equal to comparison.
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
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
template<class T, class U>
void greater_or_equal(const int m, const int n, const T* A, const int ldA,
    const U* B, const int ldB, bool* C, const int ldC);

/**
 * Less than comparison.
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
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
template<class T, class U>
void less(const int m, const int n, const T* A, const int ldA, const U* B,
    const int ldB, bool* C, const int ldC);

/**
 * Less than or equal to comparison.
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
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
template<class T, class U>
void less_or_equal(const int m, const int n, const T* A, const int ldA,
    const U* B, const int ldB, bool* C, const int ldC);

/**
 * Logical `and`.
 * 
 * @ingroup cpp-raw
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
void logical_and(const int m, const int n, const bool* A, const int ldA,
    const bool* B, const int ldB, bool* C, const int ldC);

/**
 * Logical `or`.
 * 
 * @ingroup cpp-raw
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
void logical_or(const int m, const int n, const bool* A, const int ldA,
    const bool* B, const int ldB, bool* C, const int ldC);

/**
 * Scalar multiplication.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * @tparam V Value type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param a Scalar.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
template<class T, class U, class V>
void mul(const int m, const int n, const T* a, const U* B, const int ldB,
    V* C, const int ldC);

/**
 * Matrix-vector multiplication. Computes @f$y = Ax@f$.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
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
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
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
 * Not equal comparison.
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
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
template<class T, class U>
void not_equal(const int m, const int n, const T* A, const int ldA,
    const U* B, const int ldB, bool* C, const int ldC);

/**
 * Subtraction.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * @tparam V Value type.
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
template<class T, class U, class V>
void sub(const int m, const int n, const T* A, const int ldA, const U* B,
    const int ldB, V* C, const int ldC);

}
