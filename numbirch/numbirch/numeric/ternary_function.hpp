/**
 * @file
 */
#pragma once

namespace numbirch {
/**
 * Normalized incomplete beta function.
 * 
 * @ingroup cpp-raw
 * 
 * @tparam T Floating point type.
 * 
 * @param m Number of rows.
 * @param n Number of columns.
 * @param A Matrix.
 * @param ldA Column stride of `A`.
 * @param B Matrix.
 * @param ldB Column stride of `B`.
 * @param X Matrix.
 * @param ldX Column stride of `X`.
 * @param[out] C Matrix.
 * @param ldC Column stride of `C`.
 */
template<class T>
void ibeta(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, const T* X, const int ldX, T* C, const int ldC);

}
