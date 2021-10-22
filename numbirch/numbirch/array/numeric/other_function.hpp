/**
 * @file
 */
#pragma once

#include "numbirch/array/Array.hpp"
#include "numbirch/array/Scalar.hpp"
#include "numbirch/array/Vector.hpp"
#include "numbirch/array/Matrix.hpp"
#include "numbirch/memory.hpp"
#include "numbirch/numeric.hpp"

namespace numbirch {
/**
 * Linear combination of matrices.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param a Coefficient on `A`.
 * @param A %Array.
 * @param b Coefficient on `B`.
 * @param B %Array.
 * @param c Coefficient on `C`.
 * @param C %Array.
 * @param e Coefficient on `D`.
 * @param E %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
Array<T,D> combine(const T a, const Array<T,D>& A, const T b,
    const Array<T,D>& B, const T c, const Array<T,D>& C, const T e,
    const Array<T,D>& E) {
  assert(A.rows() == B.rows());
  assert(A.rows() == C.rows());
  assert(A.rows() == E.rows());
  assert(A.columns() == B.columns());
  assert(A.columns() == C.columns());
  assert(A.columns() == E.columns());

  Array<T,D> F(A.shape().compact());
  combine(F.width(), F.height(), a, A.data(), A.stride(), b, B.data(),
      B.stride(), c, C.data(), C.stride(), e, E.data(), E.stride(), F.data(),
      F.stride());
  return F;
}

}
