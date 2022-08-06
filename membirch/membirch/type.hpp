/**
 * @file
 */
#pragma once

#include "membirch/external.hpp"

namespace membirch {
/**
 * @internal
 * 
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
 * @internal
 * 
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
 * @internal
 * 
 * Is `T` a visitable type? This is a type that provides an `accept_(V&)`
 * function.
 */
template<class T, class V>
struct is_visitable {
private:
  template<class U>
  static constexpr bool has_accept(decltype(std::declval<U>().accept_(
      std::declval<typename std::add_lvalue_reference<V>::type>()))*) {
    return true;
  }

  template<class U>
  static constexpr bool has_accept(decltype(std::declval<U>().accept_(
      std::declval<typename std::add_lvalue_reference<V>::type>(), 0, 0))*) {
    // ^ Spanner and Bridger require two additional int arguments
    return true;
  }
  template<class>
  static constexpr bool has_accept(...) {
    return false;
  }

public:
  static constexpr bool value = has_accept<T>(0);
};

/**
 * @internal
 * 
 * Is `T` an iterable type? This is a type that provides `begin()` and
 * `end()` member functions, and a `value_type` member type.
 */
template<class T>
struct is_iterable {
private:
  template<class U>
  static constexpr bool has_begin(decltype(std::declval<U>().begin())*) {
    return true;
  }
  template<class>
  static constexpr bool has_begin(...) {
    return false;
  }

  template<class U>
  static constexpr bool has_end(decltype(std::declval<U>().end())*) {
    return true;
  }
  template<class>
  static constexpr bool has_end(...) {
    return false;
  }

  template<class U>
  static constexpr bool has_value_type(typename U::value_type*) {
    return true;
  }
  template<class>
  static constexpr bool has_value_type(...) {
    return false;
  }

public:
  static constexpr bool value = has_begin<T>(0) && has_end<T>(0) &&
      has_value_type<T>(0);
};

}
