/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Less-than comparison of two objects using their possibly() function.
 */
struct possibly {
  template<class T1, class T2>
  bool operator()(T1* o1, T2* o2) {
    return o1->possibly(*o2);
  }
};
}
