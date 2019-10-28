/**
 * @file
 */
#pragma once

#include "libbirch/mutable.hpp"

namespace libbirch {
/**
 * Stride. The skip between consecutive sub-arrays along a dimension, e.g.
 * elements in a vector, rows in a matrix.
 *
 * @ingroup libbirch
 */
template<int64_t n>
struct Stride {
  static const int64_t stride_value = n;
  static const int64_t stride = n;

  Stride(const int64_t stride) {
    assert(stride == this->stride);
  }
};
template<>
struct Stride<mutable_value> {
  static const int64_t stride_value = mutable_value;
  int64_t stride;
  Stride(const int64_t stride) :
      stride(stride) {
    //
  }
};
}
