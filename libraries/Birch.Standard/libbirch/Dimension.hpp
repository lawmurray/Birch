/**
 * @file
 */
#pragma once

#include "libbirch/Length.hpp"
#include "libbirch/Stride.hpp"
#include "libbirch/Index.hpp"
#include "libbirch/Range.hpp"
#include "libbirch/Eigen.hpp"

namespace libbirch {
/**
 * Dimension of an array.
 *
 * @ingroup libbirch
 *
 * Describes one dimension of an array. It combines a Length
 * and Stride to indicate the size of the dimension. Each of
 * these may have a static value (indicated by a template argument giving
 * that value) or a dynamic value (indicated by a template argument of
 * mutable_value and initial value given in the constructor).
 */
template<int64_t length_value = mutable_value, int64_t stride_value =
    mutable_value>
struct Dimension: public Length<length_value>, public Stride<stride_value> {
  typedef Length<length_value> length_type;
  typedef Stride<stride_value> stride_type;

  /**
   * Constructor.
   *
   * @param length Initial length.
   * @param stride Initial stride.
   *
   * For static values, the initial values given must match the static values
   * or an error is given.
   */
  Dimension(const int64_t length = 0, const int64_t stride = 0) :
      length_type(length),
      stride_type(stride) {
    libbirch_assert_msg_(length >= 0,
        "dimension length is " << length << ", but must be non-negative");
  }

  /**
   * Copy constructor.
   */
  Dimension(const Dimension<length_value,stride_value>& o) = default;

  /**
   * Generic copy constructor.
   */
  template<int64_t length_value1, int64_t stride_value1>
  Dimension(const Dimension<length_value1,stride_value1>& o) :
      length_type(o.length),
      stride_type(o.stride) {
    //
  }

  /**
   * Slice operator.
   */
  template<int64_t offset_value1, int64_t length_value1>
  auto operator()(const Range<offset_value1,length_value1>& arg) const {
    static const int64_t new_length_value = length_value1;
    static const int64_t new_stride_value = stride_value;
    return Dimension<new_length_value,new_stride_value>(arg.length, this->stride);
  }

  /**
   * Generic equality operator.
   */
  template<int64_t length_value1, int64_t stride_value1>
  bool operator==(const Dimension<length_value1,stride_value1>& o) const {
    return this->length == o.length && this->stride == o.stride;
  }

  /**
   * Generic inequality operator.
   */
  template<int64_t length_value1, int64_t stride_value1>
  bool operator!=(const Dimension<length_value1,stride_value1>& o) const {
    return !(*this == o);
  }

  /**
   * Does this dimension conform to another? Two dimensions conform if their lengths
   * are equal.
   */
  template<class Dimension1>
  bool conforms(const Dimension1& o) const {
    return this->length == o.length;
  }
  bool conforms(const Eigen::Index rows) {
    return this->length == rows;
  }

  /**
   * Multiply stride.
   */
  Dimension<length_value,stride_value>& operator*=(const int64_t n) {
    /* pre-condition */
    static_assert(stride_value == mutable_value,
        "must use a mutable stride to multiply");

    this->stride *= n;

    return *this;
  }

  /**
   * Multiply stride.
   */
  auto operator*(const int64_t n) const {
    Dimension<length_value,mutable_value> result(*this);
    result *= n;
    return result;
  }
};
}
