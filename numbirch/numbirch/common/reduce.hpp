/**
 * @file
 */
#pragma once

#include "numbirch/reduce.hpp"
#include "numbirch/common/functor.hpp"

namespace numbirch {

template<class G, class T, class>
default_t<G,T> count_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(x, count_grad_functor<real>());
}

template<class G, class T, class>
default_t<G,T> sum_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(x, sum_grad_functor<real,decltype(data(g))>(data(g)));
}

}
