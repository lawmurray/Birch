/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/Offset.hpp"
#include "libbirch/Length.hpp"

namespace bi {
/**
 * Range.
 *
 * @ingroup libbirch
 *
 * A Range describes the active elements along one dimension of an array. It
 * combines an Offset and Length. Each of these may have either a
 * static value (indicated by a template parameter giving that value) or a
 * dynamic value (indicated by a template parameter of mutable_value and
 * initial value given in the constructor).
 */
template<int64_t offset_value = 0, int64_t length_value = mutable_value>
struct Range: public Offset<offset_value>, public Length<length_value> {
  typedef Offset<offset_value> offset_type;
  typedef Length<length_value> length_type;

  /**
   * Constructor.
   *
   * @param offset Initial offset.
   * @param length Initial length.
   *
   * For static values, the initial values given must match the static values
   * or an error is given.
   */
  Range(const int64_t offset, const int64_t length) :
      offset_type(offset),
      length_type(length) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<int64_t offset_value1, int64_t length_value1>
  Range(const Range<offset_value1,length_value1>& o) :
      offset_type(o.offset),
      length_type(o.length) {
    //
  }

  /**
   * Generic equality operator.
   */
  template<int64_t offset_value1, int64_t length_value1>
  bool operator==(const Range<offset_value1,length_value1>& o) const {
    return this->offset == o.offset && this->length == o.length;
  }

  /**
   * Generic inequality operator.
   */
  template<int64_t offset_value1, int64_t length_value1>
  bool operator!=(const Range<offset_value1,length_value1>& o) const {
    return !(*this == o);
  }

  /**
   * Used to count the number of ranges in a view.
   */
  static constexpr int rangeCount() {
    return 1;
  }
};
}
