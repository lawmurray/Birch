/**
 * @file
 */
#pragma once

#include "bi/lib/global.hpp"

#include <cassert>

namespace bi {
/**
 * Lead. The number of elements, including both active and inactive elements,
 * along a dimension.
 *
 * @ingroup library
 */
template<size_t n>
struct Lead {
  static const size_t lead_value = n;
  static const size_t lead = n;

  Lead(const size_t lead) {
    assert(lead == this->lead);
  }
};
template<>
struct Lead<mutable_value> {
  static const size_t lead_value = mutable_value;
  size_t lead;
  Lead(const size_t lead) :
      lead(lead) {
    //
  }
};
}
