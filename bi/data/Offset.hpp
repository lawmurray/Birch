/**
 * @file
 */
#pragma once

#include "bi/data/constant.hpp"

#include <cassert>

namespace bi {
/**
 * Offset. Number of elements until the first active element along a
 * dimension.
 *
 * @ingroup library
 */
template<int_t n>
struct Offset {
  static const int_t offset_value = n;
  static const int_t offset = n;

  Offset(const int_t offset) {
    assert(offset == this->offset);
  }
};
template<>
struct Offset<mutable_value> {
  static const int_t offset_value = mutable_value;
  int_t offset;

  Offset(const int_t offset) :
      offset(offset) {
    //
  }
};
}
