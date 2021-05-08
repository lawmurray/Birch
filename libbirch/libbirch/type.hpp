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

/**
 * If `T` is a pointer type, unwrap it to the referent type, otherwise to `T`.
 */
template<class T>
struct unwrap_pointer {
  using type = T;
};
template<class T>
struct unwrap_pointer<T&> {
  using type = typename unwrap_pointer<T>::type;
};
template<class T>
struct unwrap_pointer<T&&> {
  using type = typename unwrap_pointer<T>::type;
};

}
