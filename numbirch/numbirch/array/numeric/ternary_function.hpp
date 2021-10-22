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
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * @param X %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
Array<T,D> ibeta(const Array<T,D>& A, const Array<T,D>& B,
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

/**
 * Gradient of lchoose().
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param G %Array.
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
std::pair<Array<T,D>,Array<T,D>> lchoose_grad(const Array<T,D>& G,
    const Array<int,D>& A, const Array<int,D>& B) {
  assert(G.rows() == A.rows());
  assert(G.columns() == A.columns());
  assert(G.rows() == B.rows());
  assert(G.columns() == B.columns());

  Array<T,D> GA(A.shape().compact());
  Array<T,D> GB(A.shape().compact());
  lchoose_grad(G.width(), G.height(), G.data(), G.stride(), A.data(),
      A.stride(), B.data(), B.stride(), GA.data(), GA.stride(), GB.data(),
      GB.stride());
  return std::make_pair(GA, GB);
}

}
