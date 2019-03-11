/**
 * @file
 */
#pragma once

#include "libbirch/Shared.hpp"
#include "libbirch/Weak.hpp"
#include "libbirch/Optional.hpp"
#include "libbirch/Fiber.hpp"
#include "libbirch/Array.hpp"

namespace libbirch {
/**
 * Is this a value type?
 */
template<class T>
struct is_value {
  static const bool value = true;
};

template<class T>
struct is_value<Shared<T>> {
  static const bool value = false;
};

template<class T>
struct is_value<Weak<T>> {
  static const bool value = false;
};

template<class T>
struct is_value<Fiber<T>> {
  static const bool value = false;
};

template<class T, class F>
struct is_value<Array<T,F>> {
  static const bool value = is_value<T>::value;
};

template<class T>
struct is_value<std::initializer_list<T>> {
  static const bool value = is_value<T>::value;
};

template<class T>
struct is_value<Optional<T>> {
  static const bool value = is_value<T>::value;
};

template<class T>
struct is_value<std::function<T>> {
  static const bool value = false;
};

template<class Arg>
struct is_value<std::tuple<Arg>> {
  static const bool value = is_value<Arg>::value;
};

template<class Arg, class ... Args>
struct is_value<std::tuple<Arg,Args...>> {
  static const bool value = is_value<Arg>::value && is_value<std::tuple<Args...>>::value;
};

}
