/**
 * @file
 */
#pragma once

#include "numbirch/unary.hpp"
#include "numbirch/common/functor.hpp"

namespace numbirch {

template<class R, class T, class>
convert_t<R,T> operator+(const T& x) {
  if constexpr (std::is_same_v<R,T>) {
    return x;
  } else {
    return transform(x, identity_functor<R>());
  }
}

template<class R, class T, class>
convert_t<R,T> operator-(const T& x) {
  prefetch(x);
  return transform(x, negate_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> operator!(const T& x) {
  prefetch(x);
  return transform(x, not_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> abs(const T& x) {
  prefetch(x);
  return transform(x, abs_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> acos(const T& x) {
  prefetch(x);
  return transform(x, acos_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> asin(const T& x) {
  prefetch(x);
  return transform(x, asin_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> atan(const T& x) {
  prefetch(x);
  return transform(x, atan_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> ceil(const T& x) {
  prefetch(x);
  return transform(x, ceil_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> cos(const T& x) {
  prefetch(x);
  return transform(x, cos_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> cosh(const T& x) {
  prefetch(x);
  return transform(x, cosh_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> digamma(const T& x) {
  prefetch(x);
  return transform(x, digamma_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> exp(const T& x) {
  prefetch(x);
  return transform(x, exp_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> expm1(const T& x) {
  prefetch(x);
  return transform(x, expm1_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> floor(const T& x) {
  prefetch(x);
  return transform(x, floor_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> lfact(const T& x) {
  prefetch(x);
  return transform(x, lfact_functor<R>());
}

template<class G, class T, class>
promote_t<G,T> lfact_grad(const G& g, const T& x) {
  prefetch(x);
  return transform_grad(g, x, lfact_grad_functor<value_t<promote_t<G,T>>>());
}

template<class R, class T, class>
convert_t<R,T> lgamma(const T& x) {
  prefetch(x);
  return transform(x, lgamma_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> log(const T& x) {
  prefetch(x);
  return transform(x, log_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> log1p(const T& x) {
  prefetch(x);
  return transform(x, log1p_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> rcp(const T& x) {
  prefetch(x);
  return transform(x, rcp_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> rectify(const T& x) {
  prefetch(x);
  return transform(x, rectify_functor<R>());
}

template<class G, class T, class>
promote_t<G,T> rectify_grad(const G& g, const T& x) {
  prefetch(x);
  return transform_grad(g, x, rectify_grad_functor<
      value_t<promote_t<G,T>>>());
}

template<class R, class T, class>
convert_t<R,T> round(const T& x) {
  prefetch(x);
  return transform(x, round_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> sin(const T& x) {
  prefetch(x);
  return transform(x, sin_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> sinh(const T& x) {
  prefetch(x);
  return transform(x, sinh_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> sqrt(const T& x) {
  prefetch(x);
  return transform(x, sqrt_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> tan(const T& x) {
  prefetch(x);
  return transform(x, tan_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> tanh(const T& x) {
  prefetch(x);
  return transform(x, tanh_functor<R>());
}

}
