/**
 * @file
 */
#pragma once

#include "bi/data/Offset.hpp"
#include "bi/data/Length.hpp"
#include "bi/data/Stride.hpp"

namespace bi {
/**
 * Range.
 *
 * @ingroup library
 *
 * A Range describes the active elements along one dimension of an array. It
 * combines an Offset, Length and Stride. Each of these may have either a
 * static value (indicated by a template parameter giving that value) or a
 * dynamic value (indicated by a template parameter of mutable_value and
 * initial value given in the constructor).
 */
template<int_t offset_value = 0, int_t length_value = mutable_value,
    int_t stride_value = 1>
struct Range: public Offset<offset_value>,
    public Length<length_value>,
    public Stride<stride_value> {
  typedef Offset<offset_value> offset_type;
  typedef Length<length_value> length_type;
  typedef Stride<stride_value> stride_type;

  /**
   * Constructor.
   *
   * @param offset Initial offset.
   * @param length Initial length.
   * @param stride Initial stride.
   *
   * For static values, the initial values given must match the static values
   * or an error is given.
   */
  Range(const int_t offset = 0, const int_t length = 1,
      const int_t stride = 1) :
      offset_type(offset),
      length_type(length),
      stride_type(stride) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<int_t offset_value1, int_t length_value1, int_t stride_value1>
  Range(const Range<offset_value1,length_value1,stride_value1>& o) :
      offset_type(o.offset),
      length_type(o.length),
      stride_type(o.stride) {
    //
  }

  /**
   * Generic equality operator.
   */
  template<int_t offset_value1, int_t length_value1, int_t stride_value1>
  bool operator==(
      const Range<offset_value1,length_value1,stride_value1>& o) const {
    return this->offset == o.offset && this->length == o.length
        && this->stride == o.stride;
  }

  /**
   * Generic inequality operator.
   */
  template<int_t offset_value1, int_t length_value1, int_t stride_value1>
  bool operator!=(
      const Range<offset_value1,length_value1,stride_value1>& o) const {
    return !(*this == o);
  }
};
}
