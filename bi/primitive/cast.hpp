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
template<class Variate, class Model>
struct is_random<Random<Variate,Model>> {
  static const bool value = true;
};

/**
 * @internal
 */
template<class To, class From, bool random>
struct cast_impl {
  //
};

template<class To, class From>
struct cast_impl<To,From,false> {
  static To eval(From&& o) {
    return dynamic_cast<To>(o);
  }
};

template<class To, class From>
struct cast_impl<To,From,true> {
  static To eval(From&& o) {
    return static_cast<To>(o);
  }
};

template<class To>
struct cast_impl<To,To,true> {
  static To eval(To&& o) {
    if (o.isMissing()) {
      return o;
    } else {
      throw std::bad_cast();
    }
  }
};

/**
 * Cast object for multiple dispatch.
 */
template<class To, class From>
inline To cast(From&& o) {
  return cast_impl<To,From,is_random<typename std::decay<From>::type>::value>::eval(o);
}
}
