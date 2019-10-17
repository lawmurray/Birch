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
 * Macro that can be added to the template arguments of a class template
 * specialization to enable it only if a specific type is a value type.
 */
#define IS_VALUE(Type) class CheckType = Type, std::enable_if_t<is_value<CheckType>::value,int> = 0

/**
 * @def ARE_VALUES
 *
 * @ingroup libbirch
 *
 * Macro that can be added to the template arguments of a class template
 * specialization to enable it only if a parameter pack of types are all
 * value types.
 */
#define ARE_VALUES(Types) std::enable_if_t<is_value<Types...>::value,int> = 0

/**
 * @def IS_NOT_VALUE
 *
 * @ingroup libbirch
 *
 * Macro that can be added to the template arguments of a class template
 * specialization to enable it only if a specific type is a non-value type.
 */
#define IS_NOT_VALUE(Type) class CheckType = Type, std::enable_if_t<!is_value<CheckType>::value,int> = 0

/**
 * @def ARE_NOT_VALUES
 *
 * @ingroup libbirch
 *
 * Macro that can be added to the template arguments of a class template
 * specialization to enable it only if a parameter pack of types are all
 * not value types.
 */
#define ARE_NOT_VALUES(Types) std::enable_if_t<!is_value<Types...>::value,int> = 0

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

/**
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

/**
 * Recursively freeze objects. This is used when an object is lazily cloned,
 * to ensure that the object, and all other objects reachable from it, are
 * no longer modifiable.
 */
template<class T>
void freeze(T& o) {
  static_assert(is_value<T>::value, "unimplemented freeze()");
}

/**
 * Shallow thaw object. This is used when an object with only one remaining
 * reference is copied; instead of actually copying it is updated with a new
 * label for reuse.
 */
template<class T>
void thaw(T& o, LazyLabel* label) {
  static_assert(is_value<T>::value, "unimplemented thaw()");
}

/**
 * Recursively finish objects. This is used when an object is lazily cloned,
 * to ensure that that object, and all other objects reachable from it, are
 * no longer modifiable.
 */
template<class T>
void finish(T& o) {
  static_assert(is_value<T>::value, "unimplemented finish()");
}

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
