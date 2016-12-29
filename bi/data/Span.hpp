/**
 * @file
 */
#pragma once

#include "bi/data/Length.hpp"
#include "bi/data/Stride.hpp"
#include "bi/data/Lead.hpp"
#include "bi/data/Index.hpp"
#include "bi/data/Range.hpp"

namespace bi {
/**
 * Span.
 *
 * @ingroup library
 *
 * A Span describes one dimension of an array. It combines a Length,
 * Stride and Lead to indicate the size of the dimension. Any of
 * these may have a static value (indicated by a template parameter giving
 * that value) or a dynamic value (indicated by a template parameter of
 * mutable_value and initial value given in the constructor).
 */
template<int_t length_value = mutable_value, int_t stride_value = 1,
    int_t lead_value = mutable_value>
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
  Span(const int_t length = 1, const int_t stride = 1, const int_t lead =
      default_value) :
      length_type(length),
      stride_type(stride),
      lead_type((lead == default_value) ? stride * length : lead) {
    //
  }

  /**
   * Generic copy constructor.
   */
  template<int_t length_value1, int_t stride_value1, int_t lead_value1>
  Span(const Span<length_value1,stride_value1,lead_value1>& o) :
      length_type(o.length),
      stride_type(o.stride),
      lead_type(o.lead) {
    //
  }

  /**
   * View operator.
   */
  template<int_t other_offset_value, int_t other_length_value,
      int_t other_stride_value>
  auto operator()(
      const Range<other_offset_value,other_length_value,other_stride_value>& arg) const {
    static const int_t new_length_value = other_length_value;
    static const int_t new_stride_value = stride_value * other_stride_value;
    static const int_t new_lead_value = lead_value;
    return Span<new_length_value,new_stride_value,new_lead_value>(arg.length,
        this->stride * arg.stride, this->lead);
  }

  /**
   * Generic equality operator.
   */
  template<int_t length_value1, int_t stride_value1, int_t lead_value1>
  bool operator==(
      const Span<length_value1,stride_value1,lead_value1>& o) const {
    return this->length == o.length && this->stride == o.stride
        && this->lead == o.lead;
  }

  /**
   * Generic inequality operator.
   */
  template<int_t length_value1, int_t stride_value1, int_t lead_value1>
  bool operator!=(
      const Span<length_value1,stride_value1,lead_value1>& o) const {
    return !(*this == o);
  }

  /**
   * Does this span conform to another? Two spans conform if their lengths
   * are equal.
   */
  template<class Span1>
  bool conforms(const Span1& o) {
    return this->length == o.length;
  }

  /**
   * Multiply stride.
   */
  Span<length_value,stride_value,lead_value>& operator*=(const int_t n) {
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
  auto operator*(const int_t n) const {
    Span<length_value,mutable_value,mutable_value> result(*this);
    result *= n;
    return result;
  }
};
}
