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
 * Scalar division.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type (`double` or `float`).
 * @tparam U Value type (`double`, `float` or `int`).
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param b Scalar.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
template<class T, class U>
void div(const int m, const int n, const T* A, const int ldA, const U* b,
    T* C, const int ldC);

/**
 * Scalar multiplication.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Value type (`double`, `float` or `int`).
 * @tparam U Value type (`double` or `float`).
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param a Scalar.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
template<class T, class U>
void mul(const int m, const int n, const T* a, const U* B, const int ldB,
    U* C, const int ldC);

/**
 * Matrix-vector multiplication. Computes @f$y = Ax@f$.
 * 
 * @ingroup cpp-raw
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
 * @ingroup cpp-raw
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
 * Subtraction.
 * 
 * @ingroup cpp-raw
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

}
