/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Comparison of two pointer types. Dereferences the pointers and compares
 * their values with the less-than-or-equal operator.
 */
struct pointer_less_equal {
  template<class T1, class T2>
  bool operator()(const T1 o1, const T2 o2) {
    return *o1 <= *o2;
  }
};
}
