/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Less-than comparison of two objects using their definitely() function.
 */
struct definitely {
  template<class T1, class T2>
  bool operator()(T1* o1, T2* o2) {
    return o1->definitely(*o2);
  }
};
}
