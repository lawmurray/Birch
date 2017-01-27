/**
 * @file
 *
 * Cast functions for multiple dispatch.
 */
#pragma once

#include "bi/random/Random.hpp"

namespace bi {
/**
 * @internal
 */
template<class T>
struct is_random {
  static const bool value = false;
};

/**
 * @internal
 */
template<class Variate, class Model, class Group>
struct is_random<Random<Variate,Model,Group>> {
  static const bool value = true;
};

/**
 * @internal
 */
template<class To, class From, bool random_to, bool random_from>
struct cast_impl {
  //
};

template<class To, class From>
struct cast_impl<To,From,false,false> {
  static To eval(From&& o) {
    return dynamic_cast<To>(o);
  }
};

template<class To, class From>
struct cast_impl<To,From,true,true> {
  static To eval(From&& o) {
    if (o.isMissing()) {
      return o;
    } else {
      throw std::bad_cast();
    }
  }
};

template<class To, class From>
struct cast_impl<To,From,false,true> {
  static To eval(From&& o) {
    return static_cast<To>(o);
  }
};

/**
 * Cast object for multiple dispatch.
 */
template<class To, class From>
inline To cast(From&& o) {
  static constexpr bool random_to = is_random<typename std::decay<To>::type>::value;
  static constexpr bool random_from = is_random<typename std::decay<From>::type>::value;
  return cast_impl<To,From,random_to,random_from>::eval(o);
}
}
