/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

/**
 * @def IS_VALUE
 *
 * @ingroup libbirch
 *
 * Macro that can be added to the template arguments of a function template
 * specialization to enable it only if a specific type is a value type, using
 * SFINAE.
 */
#define IS_VALUE(Type) class CheckType1 = Type, std::enable_if_t<is_value<CheckType1>::value,int> = 0

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

template<>
struct is_value<void> {
  static const bool value = false;
};

/**
 * Is `T` a pointer type?
 */
template<class T>
struct is_pointer {
  static const bool value = false;
};

/**
 * Is `T` an array type?
 */
template<class T>
struct is_array {
  static const bool value = false;
};

/**
 * Raw pointer type corresponding to a smart pointer type `P`.
 */
template<class P>
struct raw {
  using type = void;
};

template<class T>
struct raw<T*> {
  using type = T*;
};

/**
 * Convert an object to its canonical LibBirch type. This is an identity
 * operation in most cases. For Eigen types, a overload converts objects to
 * their corresponding LibBirch type.
 *
 * This is used as a wrapper around expressions to ensure that the `auto`
 * keyword deduces the LibBirch type, and not the Eigen type, for which use
 * of `auto` can be problematic.
 *
 * SFINAE disables this implementation, it is used only for documentation.
 *
 * @see https://eigen.tuxfamily.org/dox/TopicPitfalls.html#title3
 */
template<class T, std::enable_if_t<is_value<T>::value && false,int> = 0>
auto canonical(const T& o) {
  //static_assert(false, "missing canonical() overload for this type");
  assert(false);
  return o;
}
}
