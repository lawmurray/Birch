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
 * Is `T` an acyclic type?
 *
 * An acyclic type is either:
 *
 *   - a value type, or
 *   - a pointer to an object of an acyclic class.
 *
 * @tparam T Type.
 * @tparam N Recursion bound.
 *
 * The recursion bound `N` is used to bound the number of pointer types
 * followed before concluding that a class type is cyclic. This does mean that
 * a class can be misclassified as cyclic rather than acyclic (but not the
 * other way). For the use cases of is_acyclic---such as optimizations in the
 * garbage collector---such a misclassification affects performance, but not
 * correctness.
 */
template<class T, int N = 5>
struct is_acyclic {
  static const bool value = true;
};

template<class T, int N>
struct is_acyclic<T&,N> {
  static const bool value = is_acyclic<T,N>::value;
};

template<class T, int N>
struct is_acyclic<T&&,N> {
  static const bool value = is_acyclic<T,N>::value;
};

template<int N>
struct is_acyclic<void,N> {
  static const bool value = true;
};

/**
 * Is `T` an acyclic class?
 *
 * An acyclic class is a class with all members of acyclic type.
 *
 * @seealso is_acyclic
 */
template<class T, int N = 5>
struct is_acyclic_class {
  static const bool value = is_acyclic<typename T::member_type_,N>::value;
};

}
