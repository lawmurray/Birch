/**
 * @file
 */
#pragma once

#include "bi/data/constant.hpp"

#include <cassert>

namespace bi {
/**
 * Stride. The number of elements to skip between adjacent active elements
 * along a dimension.
 *
 * @ingroup library
 */
template<int_t n>
struct Stride {
  static const int_t stride_value = n;
  static const int_t stride = n;

  Stride(const int_t stride) {
    assert(stride == this->stride);
  }
};
template<>
struct Stride<mutable_value> {
  static const int_t stride_value = mutable_value;
  int_t stride;
  Stride(const int_t stride) :
      stride(stride) {
    //
  }
};
}
