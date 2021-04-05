/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

namespace libbirch {
/**
 * Is `T` a pointer type?
 */
template<class T>
struct is_pointer {
  static const bool value = false;
};

template<class T>
struct is_pointer<T&> {
  static const bool value = is_pointer<T>::value;
};

template<class T>
struct is_pointer<T&&> {
  static const bool value = is_pointer<T>::value;
};

}
