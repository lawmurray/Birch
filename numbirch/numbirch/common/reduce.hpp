/**
 * @file
 */
#pragma once

#include "numbirch/reduce.hpp"
#include "numbirch/common/functor.hpp"

namespace numbirch {

template<class R, class T, class>
default_t<T> count_grad(const Array<real,0>& g, const Array<R,0>& y,
    const T& x) {
  prefetch(x);
  return transform(x, count_grad_functor());
}

template<class R, class T, class>
default_t<T> sum_grad(const Array<real,0>& g, const Array<R,0>& y,
    const T& x) {
  prefetch(x);
  return transform(x, sum_grad_functor<decltype(data(g))>(data(g)));
}

}
