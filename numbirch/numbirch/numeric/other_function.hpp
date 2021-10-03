/**
 * @file
 */
#pragma once

namespace numbirch {
/**
 * Linear combination of matrices.
 * 
 * @ingroup cpp-raw
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

}
