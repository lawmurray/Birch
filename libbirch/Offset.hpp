/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"

namespace bi {
/**
 * Offset. Number of elements until the first active element along a
 * dimension.
 *
 * @ingroup libbirch
 */
template<int64_t n>
struct Offset {
  static const int64_t offset_value = n;
  static const int64_t offset = n;

  Offset(const int64_t offset) {
    assert(offset == this->offset);
  }
};
template<>
struct Offset<mutable_value> {
  static const int64_t offset_value = mutable_value;
  int64_t offset;

  Offset(const int64_t offset) :
      offset(offset) {
    //
  }
};
}
