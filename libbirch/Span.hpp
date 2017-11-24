/**
 * @file
 */
#pragma once

#include "libbirch/Length.hpp"
#include "libbirch/Stride.hpp"
#include "libbirch/Lead.hpp"
#include "libbirch/Index.hpp"
#include "libbirch/Range.hpp"
#include "libbirch/Eigen.hpp"

namespace bi {
/**
 * Span.
 *
 * @ingroup libbirch
 *
 * A Span describes one dimension of an array. It combines a Length,
 * Stride and Lead to indicate the size of the dimension. Any of
 * these may have a static value (indicated by a template parameter giving
 * that value) or a dynamic value (indicated by a template parameter of
 * mutable_value and initial value given in the constructor).
 */
template<size_t length_value = mutable_value, ptrdiff_t stride_value = 1,
    size_t lead_value = mutable_value>
struct Span: public Length<length_value>,
    public Stride<stride_value>,
    public Lead<lead_value> {
  typedef Length<length_value> length_type;
  typedef Stride<stride_value> stride_type;
  typedef Lead<lead_value> lead_type;

  /**
   * Constructor.
   *
   * @param length Initial length.
   * @param stride Initial stride.
   * @param lead Initial lead.
   *
   * For static values, the initial values given must match the static values
   * or an error is given.
   */
  Span(const size_t length = 0, const ptrdiff_t stride = 1,
      const size_t lead = default_value) :
      length_type(length),
      stride_type(stride),
      lead_type((lead == default_value) ? stride * length : lead) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<size_t length_value1, ptrdiff_t stride_value1, size_t lead_value1>
  Span(const Span<length_value1,stride_value1,lead_value1>& o) :
      length_type(o.length),
      stride_type(o.stride),
      lead_type(o.lead) {
    //
  }

  /**
   * View operator.
   */
  template<ptrdiff_t other_offset_value, size_t other_length_value,
      ptrdiff_t other_stride_value>
  auto operator()(
      const Range<other_offset_value,other_length_value,other_stride_value>& arg) const {
    static const size_t new_length_value = other_length_value;
    static const ptrdiff_t new_stride_value = stride_value
        * other_stride_value;
    static const size_t new_lead_value = lead_value;
    return Span<new_length_value,new_stride_value,new_lead_value>(arg.length,
        this->stride * arg.stride, this->lead);
  }

  /**
   * Generic equality operator.
   */
  template<size_t length_value1, ptrdiff_t stride_value1, size_t lead_value1>
  bool operator==(
      const Span<length_value1,stride_value1,lead_value1>& o) const {
    return this->length == o.length && this->stride == o.stride
        && this->lead == o.lead;
  }

  /**
   * Generic inequality operator.
   */
  template<size_t length_value1, ptrdiff_t stride_value1, size_t lead_value1>
  bool operator!=(
      const Span<length_value1,stride_value1,lead_value1>& o) const {
    return !(*this == o);
  }

  /**
   * Does this span conform to another? Two spans conform if their lengths
   * are equal.
   */
  template<class Span1>
  bool conforms(const Span1& o) const {
    return this->length == o.length;
  }
  bool conforms(const Eigen::Index rows) {
    return this->length == rows;
  }

  /**
   * Resize this span to conform to another.
   */
  template<class Span1>
  void resize(const Span1& o) {
    assert(this->stride == 1);

    this->length = o.length;
    this->lead = o.length;
  }
  void resize(const Eigen::Index length) {
    assert(this->stride == 1);

    this->length = length;
    this->lead = length;
  }

  /**
   * Multiply stride.
   */
  Span<length_value,stride_value,lead_value>& operator*=(const ptrdiff_t n) {
    /* pre-condition */
    static_assert(stride_value == mutable_value, "must use a mutable stride to multiply");
    static_assert(lead_value == mutable_value, "must use a mutable lead to multiply");

    this->stride *= n;
    this->lead *= n;

    return *this;
  }

  /**
   * Multiply stride.
   */
  auto operator*(const ptrdiff_t n) const {
    Span<length_value,mutable_value,mutable_value> result(*this);
    result *= n;
    return result;
  }
};
}
