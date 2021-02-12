/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

namespace libbirch {
/**
 * Is `T` a value type?
 */
template<class T>
struct is_value {
  static const bool value = true;
};

template<class T>
struct is_value<T&> {
  static const bool value = is_value<T>::value;
};

template<class T>
struct is_value<T&&> {
  static const bool value = is_value<T>::value;
};

template<class T>
struct is_value<std::optional<T>> {
  static const bool value = is_value<T>::value;
};

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
