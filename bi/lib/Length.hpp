/**
 * @file
 */
#pragma once

#include "bi/lib/global.hpp"

#include <cassert>

namespace bi {
/**
 * Length. The number of active elements along a dimension.
 *
 * @ingroup library
 */
template<size_t n>
struct Length {
  static const size_t length_value = n;
  static const size_t length = n;

  Length(const size_t length) {
    assert(length == this->length);
  }
};
template<>
struct Length<mutable_value> {
  static const size_t length_value = mutable_value;
  size_t length;

  Length(const size_t length) :
      length(length) {
    //
  }
};
}
