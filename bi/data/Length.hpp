/**
 * @file
 */
#pragma once

#include "bi/data/constant.hpp"

#include <cassert>

namespace bi {
/**
 * Length. The number of active elements along a dimension.
 *
 * @ingroup library
 */
template<int_t n>
struct Length {
  static const int_t length_value = n;
  static const int_t length = n;

  Length(const int_t length) {
    assert(length == this->length);
  }
};
template<>
struct Length<mutable_value> {
  static const int_t length_value = mutable_value;
  int_t length;

  Length(const int_t length) :
      length(length) {
    //
  }
};
}
