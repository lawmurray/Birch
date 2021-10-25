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

#include <utility>

namespace numbirch {
/**
 * Normalized incomplete beta function.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam U Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * @param X %Array.
 * 
 * @return %Array.
 */
template<class T, class U, int D, std::enable_if_t<
    std::is_floating_point<T>::value && std::is_arithmetic<U>::value,int> = 0>
Array<T,D> ibeta(const Array<U,D>& A, const Array<U,D>& B,
    const Array<T,D>& X) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());
  assert(A.rows() == X.rows());
  assert(A.columns() == X.columns());

  Array<T,D> C(A.shape().compact());
  ibeta(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      X.data(), X.stride(), C.data(), C.stride());
  return C;
}

}
