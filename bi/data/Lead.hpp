/**
 * @file
 */
#pragma once

#include "bi/data/constant.hpp"

#include <cassert>

namespace bi {
/**
 * Lead. The number of elements, including both active and inactive elements,
 * along a dimension.
 *
 * @ingroup library
 */
template<int_t n>
struct Lead {
  static const int_t lead_value = n;
  static const int_t lead = n;

  Lead(const int_t lead) {
    assert(lead == this->lead);
  }
};
template<>
struct Lead<mutable_value> {
  static const int_t lead_value = mutable_value;
  int_t lead;
  Lead(const int_t lead) :
      lead(lead) {
    //
  }
};
}
