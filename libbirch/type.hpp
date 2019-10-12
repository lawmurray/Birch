/**
 * @file
 */
#pragma once

/**
 * @def IS_VALUE
 *
 * Macro that can be added to template parameters to enable a function only
 * if the specific type is a value type.
 */
#define IS_VALUE(Type) class T = Type, std::enable_if_t<is_value<T>::value,int> = 0

/**
 * @def IS_NOT_VALUE
 *
 * Macro that can be added to template parameters to enable a function only
 * if the specific type is not a value type.
 */
#define IS_NOT_VALUE(Type) class T = Type, std::enable_if_t<!is_value<T>::value,int> = 0

namespace libbirch {
/*
 * Is this a value type?
 */
template<class T>
struct is_value {
  static const bool value = true;
};

/*
 * Is this a pointer type?
 */
template<class T>
struct is_pointer {
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
