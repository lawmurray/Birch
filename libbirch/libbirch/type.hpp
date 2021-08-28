/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

namespace libbirch {
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
 * @internal
 * 
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
