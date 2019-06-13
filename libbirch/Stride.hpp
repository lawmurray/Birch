/**
 * @file
 */
#pragma once

#include "libbirch/mutable.hpp"

namespace libbirch {
#pragma omp declare target
/**
 * Stride. The number of elements, including both active and inactive elements,
 * along a dimension.
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
#pragma omp end declare target
}
