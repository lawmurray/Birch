/**
 * @file
 */
#pragma once

#include "bi/data/Range.hpp"
#include "bi/data/View.hpp"
#include "bi/data/constant.hpp"

namespace bi {
/**
 * Static computation of the greatest common divisor of two positive
 * integers. If either is zero, returns zero.
 */
template<int_t a, int_t b>
struct gcd {
  static const int_t value = gcd<b,a % b>::value;
};
template<int_t a>
struct gcd<a,a> {
  static const int_t value = a;
};
template<int_t a>
struct gcd<a,0> {
  static const int_t value = a;
};
template<int_t b>
struct gcd<0,b> {
  static const int_t value = b;
};
template<>
struct gcd<0,0> {
  static const int_t value = 0;
};

/**
 * Greatest common divisor of two positive integers.
 */
int_t common_length(const int_t a, const int_t b);

/**
 * Range with length that is the greatest common divisor of the lengths of
 * two other ranges.
 */
template<int_t offset_value1, int_t length_value1, int_t stride_value1,
    int_t offset_value2, int_t length_value2, int_t stride_value2>
auto common_range(const Range<offset_value1,length_value1,stride_value1>& o1,
    const Range<offset_value2,length_value2,stride_value2>& o2) {
  /* pre-condition */
  assert(o1.stride == 1);
  assert(o2.stride == 1);

  static const int_t offset_value = 0;
  static const int_t length_value = gcd<length_value1,length_value2>::value;
  static const int_t stride_value = 1;

  const int_t offset = 0;
  const int_t length = common_length(o1.length, o2.length);
  const int_t stride = 1;

  return Range<offset_value,length_value,stride_value>(offset, length, stride);
}

/**
 * Frame with span lengths that are the greatest common divisor of the
 * lengths of the spans of two other frames.
 */
template<class View1, class View2>
auto common_view(const View1& o1, const View2& o2) {
  auto tail = common_view(o1.tail, o2.tail);
  auto head = common_range(o1.head, o2.head);

  return NonemptyView<decltype(tail),decltype(head)>(tail, head);
}
inline EmptyView common_view(const EmptyView& o1, const EmptyView& o2) {
  return EmptyView();
}

}
