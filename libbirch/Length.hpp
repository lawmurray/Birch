/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/global.hpp"

namespace bi {
/**
 * Length. The number of active elements along a dimension.
 *
 * @ingroup libbirch
 */
template<int64_t n>
struct Length {
  static const int64_t length_value = n;
  static const int64_t length = n;

  Length(const int64_t length) {
    assert(length == this->length);
  }
};
template<>
struct Length<mutable_value> {
  static const int64_t length_value = mutable_value;
  int64_t length;

  Length(const int64_t length) :
      length(length) {
    //
  }
};
}
