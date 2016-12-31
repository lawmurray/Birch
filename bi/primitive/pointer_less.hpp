/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Comparison of two pointer types. Dereferences the pointers and compares
 * their values with the less-than operator.
 */
struct pointer_less {
  template<class T1, class T2>
  bool operator()(T1* o1, T2* o2) const {
    return *o1 < *o2;
  }
};
}
