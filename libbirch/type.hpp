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

/**
 * @def IS_CONVERTIBLE
 *
 * @ingroup libbirch
 *
 * Macro that can be added to the template arguments of a function template
 * specialization to enable it only if one type is convertible to another,
 * using SFINAE.
 */
#define IS_CONVERTIBLE(From,To) std::enable_if_t<std::is_convertible<From,To>::value,int> = 0

/**
 * @def IS_DEFAULT_CONSTRUCTIBLE
 *
 * Macro that can be added to the template arguments of a function template
 * specialization to enable it only if a specific type is
 * default-constructible using SFINAE.
 */
#define IS_DEFAULT_CONSTRUCTIBLE(Type) std::enable_if_t<is_pointer<Type>::value && std::is_constructible<typename Type::value_type,Label*>::value,int> = 0

/**
 * @def IS_NOT_DEFAULT_CONSTRUCTIBLE
 *
 * Macro that can be added to the template arguments of a function template
 * specialization to enable it only if a specific type is not
 * default-constructible using SFINAE.
 */
#define IS_NOT_DEFAULT_CONSTRUCTIBLE(Type) std::enable_if_t<is_pointer<Type>::value && !std::is_constructible<typename Type::value_type,Label*>::value,int> = 0

/**
 * @def IS_VOID
 *
 * Macro that can be added to the template arguments of a class template
 * specialization to enable it only if a specific type is void, using SFINAE.
 */
#define IS_VOID(Type) std::enable_if_t<std::is_void<Type>::value>

/**
 * @def IS_NOT_VOID
 *
 * Macro that can be added to the template arguments of a class template
 * specialization to enable it only if a specific type is not void, using
 * SFINAE.
 */
#define IS_NOT_VOID(Type) std::enable_if_t<!std::is_void<Type>::value>

/**
 * @def IS_SAME
 *
 * Macro that can be added to the template arguments of a function template
 * specialization to enable it only if a pair of types are the same,
 * using SFINAE.
 */
#define IS_SAME(Type1,Type2) std::enable_if_t<std::is_same<Type1,Type2>::value,int> = 0

/**
 * @def IS_NOT_SAME
 *
 * Macro that can be added to the template arguments of a function template
 * specialization to enable it only if a pair of types are not the same,
 * using SFINAE.
 */
#define IS_NOT_SAME(Type1,Type2) std::enable_if_t<!std::is_same<Type1,Type2>::value,int> = 0

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
 * @see https://eigen.tuxfamily.org/dox/TopicPitfalls.html#title3
 */
template<class T>
auto canonical(T&& o) {
  return std::forward<T>(o);
}
}
