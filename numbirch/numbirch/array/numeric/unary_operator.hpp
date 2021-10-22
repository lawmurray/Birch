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
 * Logical `not`.
 * 
 * @ingroup array
 * 
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<int D>
Array<bool,D> operator!(const Array<bool,D>& A) {
  Array<bool,D> B(A.shape().compact());
  logical_not(A.width(), A.height(), A.data(), A.stride(), B.data(),
      B.stride());
  return B;
}

/**
 * Negation.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_arithmetic<T>::value,int> = 0>
Array<T,D> operator-(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  neg(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

}
