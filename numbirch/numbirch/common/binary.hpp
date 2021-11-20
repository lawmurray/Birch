/**
 * @file
 */
#pragma once

#include "numbirch/binary.hpp"
#include "numbirch/common/functor.hpp"

namespace numbirch {

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> operator+(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, add_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> operator-(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, subtract_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> operator*(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, multiply_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> operator/(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, divide_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> operator&&(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, and_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> operator||(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, or_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> operator==(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, equal_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> operator!=(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, not_equal_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> operator<(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, less_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> operator<=(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, less_or_equal_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> operator>(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, greater_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> operator>=(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, greater_or_equal_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> copysign(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, copysign_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> digamma(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, digamma_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> gamma_p(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, gamma_p_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> gamma_q(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, gamma_q_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> hadamard(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, multiply_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> lbeta(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, lbeta_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> lchoose(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, lchoose_functor<R>());
}

template<class G, class T, class U, class>
std::pair<implicit_t<G,T,U>,implicit_t<G,T,U>> lchoose_grad(const G& g,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_grad(g, x, y,
      lchoose_grad_functor<value_t<implicit_t<G,T,U>>>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> lgamma(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, lgamma_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,implicit_t<T,U>> pow(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, pow_functor<R>());
}

}
