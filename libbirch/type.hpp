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
template<class T, unsigned N = 5>
struct is_acyclic {
  static const bool value = true;
};

template<class T, unsigned N>
struct is_acyclic<T&,N> {
  static const bool value = is_acyclic<T,N>::value;
};

template<class T, unsigned N>
struct is_acyclic<T&&,N> {
  static const bool value = is_acyclic<T,N>::value;
};

template<unsigned N>
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
template<class T, unsigned N = 5>
struct is_acyclic_class {
  static const bool value = is_acyclic<typename T::member_type_,N>::value;
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
template<class T, std::enable_if_t<std::is_void<T>::value,int> = 0>
auto canonical(const T& o) {
  assert(false);
  return o;
}
}
