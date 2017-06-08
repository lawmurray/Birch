/**
 * @file
 */
#pragma once

#include "bi/lib/Offset.hpp"

namespace bi {
/**
 * Index.
 *
 * @ingroup library
 *
 * An Index indicates one element along one dimension of an array.
 */
template<ptrdiff_t offset_value = 0>
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
  Index(const ptrdiff_t offset = 0) :
      offset_type(offset), length_type(1) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<ptrdiff_t offset_value1>
  Index(const Index<offset_value1>& o) :
      offset_type(o.offset), length_type(o.length) {
    //
  }

  /**
   * Generic equality operator. Two spans are considered equal if they have
   * the same offset and length.
   */
  template<ptrdiff_t offset_value1>
  bool operator==(const Index<offset_value1>& o) const {
    return this->offset == o.offset;
  }

  /**
   * Generic inequality operator.
   */
  template<ptrdiff_t offset_value1>
  bool operator!=(const Index<offset_value1>& o) const {
    return !(*this == o);
  }
};
}
