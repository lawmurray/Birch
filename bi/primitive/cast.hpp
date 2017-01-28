/**
 * @file
 *
 * Cast functions for multiple dispatch.
 */
#pragma once

#include "bi/random/Random.hpp"
#include "bi/data/MemoryPrimitiveValue.hpp"

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

/**
 * Cast to derived type.
 */
template<class To, class From>
struct cast_impl<To,From,false,false> {
  static To eval(From&& o) {
    return dynamic_cast<To>(o);
  }
};

/**
 * Cast to built-in type.
 */
template<class From>
struct cast_impl<const double&,From,false,false> {
  static const double& eval(From&& o) {
    return static_cast<const double&>(o);
  }
};

/**
 * Cast random to random (probably to the same type, but need to check if the
 * random variable is eligible).
 */
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

/**
 * Cast random to non-random.
 */
template<class To, class From>
struct cast_impl<To,From,false,true> {
  static To eval(From&& o) {
    return static_cast<To>(o);
  }
};

/**
 * Cast random to built-in type.
 */
template<class From>
struct cast_impl<const double&,From,false,true> {
  static const double& eval(From&& o) {
    typedef typename std::decay<From>::type::group_type group_type;
    return static_cast<const double&>(static_cast<PrimitiveValue<double,group_type>>(o));
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
