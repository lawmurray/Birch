/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Comparison of two objects by their definitely() function.
 */
struct definitely {
  template<class T1, class T2>
  bool operator()(T1* o1, T2* o2) {
    return o1->definitely(*o2);
  }
};
}
