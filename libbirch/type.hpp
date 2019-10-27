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
#define IS_VALUE1(Type) class CheckType1 = Type, std::enable_if_t<is_value<CheckType1>::value,int> = 0

/**
 * @def IS_VALUE2
 *
 * @ingroup libbirch
 *
 * As IS_VALUE1, for use when a second condition is required.
 */
#define IS_VALUE2(Type) class CheckType2 = Type, std::enable_if_t<is_value<CheckType2>::value,int> = 0

/**
 * @def IS_VALUE
 *
 * As IS_VALUE1.
 */
#define IS_VALUE(Type) IS_VALUE1(Type)

/**
 * @def IS_NOT_VALUE1
 *
 * @ingroup libbirch
 *
 * Macro that can be added to the template arguments of a function template
 * specialization to enable it only if a specific type is a non-value type,
 * using SFINAE.
 */
#define IS_NOT_VALUE1(Type) class CheckType1 = Type, std::enable_if_t<!is_value<CheckType1>::value,int> = 0

/**
 * @def IS_NOT_VALUE2
 *
 * @ingroup libbirch
 *
 * As IS_NOT_VALUE1, for use when a second condition is required.
 */
#define IS_NOT_VALUE2(Type) class CheckType2 = Type, std::enable_if_t<!is_value<CheckType2>::value,int> = 0

/**
 * @def IS_NOT_VALUE
 *
 * As IS_NOT_VALUE1.
 */
#define IS_NOT_VALUE(Type) IS_NOT_VALUE1(Type)

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
 * @def IS_POINTER
 *
 * Macro that can be added to the template arguments of a class template
 * specialization to enable it only if a specific type is a pointer type,
 * using SFINAE.
 */
#define IS_POINTER(Type) std::enable_if_t<is_pointer<Type>::value>

/**
 * @def IS_NOT_POINTER
 *
 * Macro that can be added to the template arguments of a class template
 * specialization to enable it only if a specific type is not a pointer type,
 * using SFINAE.
 */
#define IS_NOT_POINTER(Type) std::enable_if_t<!is_pointer<Type>::value>

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

namespace libbirch {
/*
 * Are these value types?
 */
template<class Arg, class... Args>
struct is_value {
  static const bool value = is_value<Arg>::value && is_value<Args...>::value;
};

/*
 * Is this a value type?
 */
template<class Arg>
struct is_value<Arg> {
  static const bool value = true;
};

/*
 * Are these pointer types?
 */
template<class Arg, class... Args>
struct is_pointer {
  static const bool value = is_pointer<Arg>::value && is_pointer<Args...>::value;
};

/*
 * Is this a pointer type?
 */
template<class Arg>
struct is_pointer<Arg> {
  static const bool value = false;
};

template<class T>
struct is_value<std::function<T>> {
  static const bool value = false;
};

template<class T>
void freeze(std::function<T>& o) {
  assert(false);
  /// @todo Need to freeze any objects in the closure here, which may require
  /// a custom implementation of lambda functions in a similar way to fibers,
  /// rather than using std::function
}

template<class T>
void thaw(std::function<T>& o) {
  assert(false);
  /// @todo Need to thaw any objects in the closure here, which may require
  /// a custom implementation of lambda functions in a similar way to fibers,
  /// rather than using std::function
}

template<class T>
void finish(std::function<T>& o) {
  assert(false);
  /// @todo Need to finish any objects in the closure here, which may require
  /// a custom implementation of lambda functions in a similar way to fibers,
  /// rather than using std::function
}

}
