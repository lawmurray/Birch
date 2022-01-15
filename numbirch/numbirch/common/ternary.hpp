/**
 * @file
 */
#pragma once

#include "numbirch/ternary.hpp"
#include "numbirch/common/functor.hpp"

namespace numbirch {

template<class R, class T, class U, class V, class>
explicit_t<R,T,U,V> ibeta(const T& x, const U& y, const V& z) {
  prefetch(x);
  prefetch(y);
  prefetch(z);
  return transform(x, y, z, ibeta_functor());
}

}
