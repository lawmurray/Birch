/**
 * @file
 *
 * Cast functions for multiple dispatch.
 */
#pragma once

#include "bi/method/Random.hpp"
#include "bi/data/MemoryPrimitiveValue.hpp"

namespace bi {
enum TypeFlag {
  IS_PRIMITIVE, IS_MODEL, IS_RANDOM, IS_LAMBDA
};

template<class T>
struct type_flag {
  static const TypeFlag value =
      std::is_class<T>::value ? IS_MODEL : IS_PRIMITIVE;
};

template<class Value, class Distribution>
struct type_flag<Random<Value,Distribution>> {
  static const TypeFlag value = IS_RANDOM;
};

template<class Result>
struct type_flag<Lambda<Result>> {
  static const TypeFlag value = IS_LAMBDA;
};

/**
 * Cast.
 */
template<class To, class From, TypeFlag to_flag, TypeFlag from_flag>
struct cast_impl {
  static To eval(From&& o) {
    throw std::bad_cast();
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_PRIMITIVE,IS_PRIMITIVE> {
  static To eval(From&& o) {
    return static_cast<To>(o);
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_PRIMITIVE,IS_MODEL> {
  static To eval(From&& o) {
    throw std::bad_cast();
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_PRIMITIVE,IS_RANDOM> {
  static To eval(From&& o) {
    return static_cast<To>(o);
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_PRIMITIVE,IS_LAMBDA> {
  static To eval(From&& o) {
    throw std::bad_cast();
    //return static_cast<To>(o);
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_MODEL,IS_PRIMITIVE> {
  static To eval(From&& o) {
    throw std::bad_cast();
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_MODEL,IS_MODEL> {
  static To eval(From&& o) {
    return dynamic_cast<To>(o);
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_MODEL,IS_RANDOM> {
  static To eval(From&& o) {
    return static_cast<To>(o);
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_MODEL,IS_LAMBDA> {
  static To eval(From&& o) {
    return static_cast<To>(o);
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_RANDOM,IS_PRIMITIVE> {
  static To eval(From&& o) {
    throw std::bad_cast();
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_RANDOM,IS_MODEL> {
  static To eval(From&& o) {
    throw std::bad_cast();
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_RANDOM,IS_RANDOM> {
  static To eval(From&& o) {
    if (o.state == MISSING) {
      return dynamic_cast<To>(o);
    } else {
      throw std::bad_cast();
    }
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_RANDOM,IS_LAMBDA> {
  static To eval(From&& o) {
    throw std::bad_cast();
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_LAMBDA,IS_PRIMITIVE> {
  static To eval(From&& o) {
    throw std::bad_cast();
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_LAMBDA,IS_MODEL> {
  static To eval(From&& o) {
    throw std::bad_cast();
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_LAMBDA,IS_RANDOM> {
  static To eval(From&& o) {
    throw std::bad_cast();
  }
};

template<class To, class From>
struct cast_impl<To,From,IS_LAMBDA,IS_LAMBDA> {
  static To eval(From&& o) {
    throw std::bad_cast();
  }
};

/**
 * Cast.
 */
template<class To, class From>
inline To cast(From&& o) {
  static constexpr TypeFlag to_flag =
      type_flag<typename std::decay<To>::type>::value;
  static constexpr TypeFlag from_flag = type_flag<
      typename std::decay<From>::type>::value;
  return cast_impl<To,From,to_flag,from_flag>::eval(o);
}
}
