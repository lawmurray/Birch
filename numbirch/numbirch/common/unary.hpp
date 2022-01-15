/**
 * @file
 */
#pragma once

#include "numbirch/unary.hpp"
#include "numbirch/common/functor.hpp"

namespace numbirch {

template<class T, class>
T operator-(const T& x) {
  prefetch(x);
  return transform(x, negate_functor());
}

template<class T, class>
explicit_t<bool,T> operator!(const T& x) {
  prefetch(x);
  return transform(x, not_functor());
}

template<class T, class>
default_t<T> not_grad(const default_t<T>& g, const explicit_t<bool,T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, not_grad_functor());
}

template<class T, class>
T abs(const T& x) {
  prefetch(x);
  return transform(x, abs_functor());
}

template<class T, class>
default_t<T> abs_grad(const default_t<T>& g, const T& y, const T& x) {
  prefetch(x);
  return transform(g, x, abs_grad_functor());
}

template<class T, class>
default_t<T> acos(const T& x) {
  prefetch(x);
  return transform(x, acos_functor());
}

template<class T, class>
default_t<T> acos_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, acos_grad_functor());
}

template<class T, class>
default_t<T> asin(const T& x) {
  prefetch(x);
  return transform(x, asin_functor());
}

template<class T, class>
default_t<T> asin_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, asin_grad_functor());
}

template<class T, class>
default_t<T> atan(const T& x) {
  prefetch(x);
  return transform(x, atan_functor());
}

template<class T, class>
default_t<T> atan_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, atan_grad_functor());
}

template<class R, class T, class>
explicit_t<R,T> cast(const T& x) {
  prefetch(x);
  return transform(x, cast_functor<R>());
}

template<class T, class>
T ceil(const T& x) {
  prefetch(x);
  return transform(x, ceil_functor());
}

template<class T, class>
default_t<T> ceil_grad(const default_t<T>& g, const T& y, const T& x) {
  prefetch(x);
  return transform(g, x, ceil_grad_functor());
}

template<class T, class>
default_t<T> cos(const T& x) {
  prefetch(x);
  return transform(x, cos_functor());
}

template<class T, class>
default_t<T> cos_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, cos_grad_functor());
}

template<class T, class>
default_t<T> cosh(const T& x) {
  prefetch(x);
  return transform(x, cosh_functor());
}

template<class T, class>
default_t<T> cosh_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, cosh_grad_functor());
}

template<class T, class>
default_t<T> digamma(const T& x) {
  prefetch(x);
  return transform(x, digamma_functor());
}

template<class T, class>
default_t<T> exp(const T& x) {
  prefetch(x);
  return transform(x, exp_functor());
}

template<class T, class>
default_t<T> expm1(const T& x) {
  prefetch(x);
  return transform(x, expm1_functor());
}

template<class T, class>
T floor(const T& x) {
  prefetch(x);
  return transform(x, floor_functor());
}

template<class T, class>
default_t<T> floor_grad(const default_t<T>& g, const T& y, const T& x) {
  prefetch(x);
  return transform(g, x, floor_grad_functor());
}

template<class T, class>
default_t<T> lfact(const T& x) {
  prefetch(x);
  return transform(x, lfact_functor());
}

template<class T, class>
default_t<T> lfact_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, lfact_grad_functor());
}

template<class T, class>
default_t<T> lgamma(const T& x) {
  prefetch(x);
  return transform(x, lgamma_functor());
}

template<class T, class>
default_t<T> lgamma_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x,
      lgamma_grad_functor());
}

template<class T, class>
default_t<T> log(const T& x) {
  prefetch(x);
  return transform(x, log_functor());
}

template<class T, class>
default_t<T> log_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, log_grad_functor());
}

template<class T, class>
default_t<T> log1p(const T& x) {
  prefetch(x);
  return transform(x, log1p_functor());
}

template<class T, class>
default_t<T> log1p_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, log1p_grad_functor());
}

template<class T, class>
T rectify(const T& x) {
  prefetch(x);
  return transform(x, rectify_functor());
}

template<class T, class>
default_t<T> rectify_grad(const default_t<T>& g, const T& y, const T& x) {
  prefetch(x);
  return transform(g, x, rectify_grad_functor());
}

template<class T, class>
T round(const T& x) {
  prefetch(x);
  return transform(x, round_functor());
}

template<class T, class>
default_t<T> round_grad(const default_t<T>& g, const T& y, const T& x) {
  prefetch(x);
  return transform(g, x, round_grad_functor());
}

template<class T, class>
default_t<T> sin(const T& x) {
  prefetch(x);
  return transform(x, sin_functor());
}

template<class T, class>
default_t<T> sin_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, sin_grad_functor());
}

template<class T, class>
default_t<T> sinh(const T& x) {
  prefetch(x);
  return transform(x, sinh_functor());
}

template<class T, class>
default_t<T> sinh_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, sinh_grad_functor());
}

template<class T, class>
default_t<T> sqrt(const T& x) {
  prefetch(x);
  return transform(x, sqrt_functor());
}

template<class T, class>
default_t<T> sqrt_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, sqrt_grad_functor());
}

template<class T, class>
default_t<T> tan(const T& x) {
  prefetch(x);
  return transform(x, tan_functor());
}

template<class T, class>
default_t<T> tan_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, tan_grad_functor());
}

template<class T, class>
default_t<T> tanh(const T& x) {
  prefetch(x);
  return transform(x, tanh_functor());
}

template<class T, class>
default_t<T> tanh_grad(const default_t<T>& g, const default_t<T>& y,
    const T& x) {
  prefetch(x);
  return transform(g, x, tanh_grad_functor());
}

}
