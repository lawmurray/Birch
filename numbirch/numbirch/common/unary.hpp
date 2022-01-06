/**
 * @file
 */
#pragma once

#include "numbirch/unary.hpp"
#include "numbirch/common/functor.hpp"

namespace numbirch {

template<class R, class T, class>
explicit_t<R,T> operator+(const T& x) {
  if constexpr (std::is_same_v<R,value_t<T>>) {
    return x;
  } else {
    prefetch(x);
    return transform(x, cast_functor<R>());
  }
}

template<class R, class T, class>
explicit_t<R,T> operator-(const T& x) {
  prefetch(x);
  return transform(x, negate_functor<R>());
}

template<class R, class T, class>
explicit_t<R,T> operator!(const T& x) {
  prefetch(x);
  return transform(x, not_functor<R>());
}

template<class G, class T, class>
default_t<G,T> not_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, not_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> abs(const T& x) {
  if constexpr (std::is_signed_v<R>) {
    prefetch(x);
    return transform(x, abs_functor<R>());
  } else {
    return x;
  }
}

template<class G, class T, class>
default_t<G,T> abs_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, abs_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> acos(const T& x) {
  prefetch(x);
  return transform(x, acos_functor<R>());
}

template<class G, class T, class>
default_t<G,T> acos_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, acos_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> asin(const T& x) {
  prefetch(x);
  return transform(x, asin_functor<R>());
}

template<class G, class T, class>
default_t<G,T> asin_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, asin_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> atan(const T& x) {
  prefetch(x);
  return transform(x, atan_functor<R>());
}

template<class G, class T, class>
default_t<G,T> atan_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, atan_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> cast(const T& x) {
  if constexpr (std::is_same_v<R,value_t<T>>) {
    return x;
  } else {
    prefetch(x);
    return transform(x, cast_functor<R>());
  }
}

template<class R, class T, class>
explicit_t<R,T> ceil(const T& x) {
  prefetch(x);
  return transform(x, ceil_functor<R>());
}

template<class G, class T, class>
default_t<G,T> ceil_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, ceil_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> cos(const T& x) {
  prefetch(x);
  return transform(x, cos_functor<R>());
}

template<class G, class T, class>
default_t<G,T> cos_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, cos_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> cosh(const T& x) {
  prefetch(x);
  return transform(x, cosh_functor<R>());
}

template<class G, class T, class>
default_t<G,T> cosh_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, cosh_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> digamma(const T& x) {
  prefetch(x);
  return transform(x, digamma_functor<R>());
}

template<class R, class T, class>
explicit_t<R,T> exp(const T& x) {
  prefetch(x);
  return transform(x, exp_functor<R>());
}

template<class G, class T, class>
default_t<G,T> exp_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, exp_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> expm1(const T& x) {
  prefetch(x);
  return transform(x, expm1_functor<R>());
}

template<class G, class T, class>
default_t<G,T> expm1_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, expm1_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> floor(const T& x) {
  prefetch(x);
  return transform(x, floor_functor<R>());
}

template<class G, class T, class>
default_t<G,T> floor_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, floor_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> lfact(const T& x) {
  prefetch(x);
  return transform(x, lfact_functor<R>());
}

template<class G, class T, class>
default_t<G,T> lfact_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, lfact_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> lgamma(const T& x) {
  prefetch(x);
  return transform(x, lgamma_functor<R>());
}

template<class G, class T, class>
default_t<G,T> lgamma_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x,
      lgamma_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> log(const T& x) {
  prefetch(x);
  return transform(x, log_functor<R>());
}

template<class G, class T, class>
default_t<G,T> log_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, log_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> log1p(const T& x) {
  prefetch(x);
  return transform(x, log1p_functor<R>());
}

template<class G, class T, class>
default_t<G,T> log1p_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, log1p_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> rectify(const T& x) {
  prefetch(x);
  return transform(x, rectify_functor<R>());
}

template<class G, class T, class>
default_t<G,T> rectify_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, rectify_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> round(const T& x) {
  prefetch(x);
  return transform(x, round_functor<R>());
}

template<class G, class T, class>
default_t<G,T> round_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, round_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> sin(const T& x) {
  prefetch(x);
  return transform(x, sin_functor<R>());
}

template<class G, class T, class>
default_t<G,T> sin_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, sin_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> sinh(const T& x) {
  prefetch(x);
  return transform(x, sinh_functor<R>());
}

template<class G, class T, class>
default_t<G,T> sinh_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, sinh_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> sqrt(const T& x) {
  prefetch(x);
  return transform(x, sqrt_functor<R>());
}

template<class G, class T, class>
default_t<G,T> sqrt_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, sqrt_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> tan(const T& x) {
  prefetch(x);
  return transform(x, tan_functor<R>());
}

template<class G, class T, class>
default_t<G,T> tan_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, tan_grad_functor<real>());
}

template<class R, class T, class>
explicit_t<R,T> tanh(const T& x) {
  prefetch(x);
  return transform(x, tanh_functor<R>());
}

template<class G, class T, class>
default_t<G,T> tanh_grad(const G& g, const T& x) {
  prefetch(x);
  return transform(g, x, tanh_grad_functor<real>());
}

}
