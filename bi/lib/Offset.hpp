/**
 * @file
 */
#pragma once

#include "bi/lib/constant.hpp"

#include <cassert>

namespace bi {
/**
 * Offset. Number of elements until the first active element along a
 * dimension.
 *
 * @ingroup library
 */
template<ptrdiff_t n>
struct Offset {
  static const ptrdiff_t offset_value = n;
  static const ptrdiff_t offset = n;

  Offset(const ptrdiff_t offset) {
    assert(offset == this->offset);
  }
};
template<>
struct Offset<mutable_value> {
  static const ptrdiff_t offset_value = mutable_value;
  ptrdiff_t offset;

  Offset(const ptrdiff_t offset) :
      offset(offset) {
    //
  }
};
}
