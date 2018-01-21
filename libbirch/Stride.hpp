/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

#include <cassert>

namespace bi {
/**
 * Stride. The number of elements, including both active and inactive elements,
 * along a dimension.
 *
 * @ingroup libbirch
 */
template<size_t n>
struct Stride {
  static const size_t stride_value = n;
  static const size_t stride = n;

  Stride(const size_t stride) {
    assert(stride == this->stride);
  }
};
template<>
struct Stride<mutable_value> {
  static const size_t stride_value = mutable_value;
  size_t stride;
  Stride(const size_t stride) :
      stride(stride) {
    //
  }
};
}
