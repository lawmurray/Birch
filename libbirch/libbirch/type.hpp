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

/**
 * Is `T` an inplace type?
 */
template<class T>
struct is_inplace {
  static const bool value = false;
};

template<class T>
struct is_inplace<T&> {
  static const bool value = is_inplace<T>::value;
};

template<class T>
struct is_inplace<T&&> {
  static const bool value = is_inplace<T>::value;
};

/**
 * If `T` is an inplace type, unwrap it to the referent type, otherwise to
 * `T`.
 */
template<class T>
struct unwrap_inplace {
  using type = T;
};
template<class T>
struct unwrap_inplace<T&> {
  using type = typename unwrap_inplace<T>::type;
};
template<class T>
struct unwrap_inplace<T&&> {
  using type = typename unwrap_inplace<T>::type;
};

}
