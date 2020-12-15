/**
 * @file
 */
#pragma once

#include "libbirch/Offset.hpp"

namespace libbirch {
/**
 * Index, used within a slice to indicate a single element of a dimension.
 *
 * @ingroup libbirch
 *
 * An Index indicates one element along one dimension of an array. It
 * consists of an Offset only, which may have either a static value
 * (indicated by a template argument giving that value) or a dynamic value
 * (indicated by a template argument of mutable_value and initial value given
 * in the constructor).
 */
template<int64_t offset_value = 0>
struct Index: public Offset<offset_value>, public Length<1> {
  typedef Offset<offset_value> offset_type;
  typedef Length<1> length_type;

  /**
   * Constructor.
   *
   * @param offset Initial offset.
   *
   * For static values, the initial values given must match the static values
   * or an error is given.
   */
  Index(const int64_t offset = 0) :
      offset_type(offset), length_type(1) {
    //
  }

  /**
   * Copy constructor.
   */
  Index(const Index<offset_value>& o) = default;

  /**
   * Generic copy constructor.
   */
  template<int64_t offset_value1>
  Index(const Index<offset_value1>& o) :
      offset_type(o.offset), length_type(o.length) {
    //
  }

  /**
   * Generic equality operator.
   */
  template<int64_t offset_value1>
  bool operator==(const Index<offset_value1>& o) const {
    return this->offset == o.offset;
  }

  /**
   * Generic inequality operator.
   */
  template<int64_t offset_value1>
  bool operator!=(const Index<offset_value1>& o) const {
    return !(*this == o);
  }

  /**
   * Used to count the number of ranges in a slice.
   */
  static constexpr int rangeCount() {
    return 0;
  }
};

/**
 * Make an index.
 *
 * @ingroup libbirch
 *
 * @param offset Index, 1-based.
 */
inline Index<> make_index(const int64_t offset) {
  return Index<>(offset - 1);
}

}
