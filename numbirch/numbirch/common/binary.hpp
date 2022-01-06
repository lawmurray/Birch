/**
 * @file
 */
#pragma once

#include "numbirch/binary.hpp"
#include "numbirch/common/functor.hpp"

namespace numbirch {

template<class R, class T, class U, class>
explicit_t<R,T,U> operator+(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, add_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> operator-(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, subtract_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> operator*(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  if constexpr (is_scalar_v<U>) {
    return transform(x, scalar_multiply_functor<R,decltype(data(y))>(data(y)));
  } else {
    return transform(y, scalar_multiply_functor<R,decltype(data(x))>(data(x)));
  }
}

template<class R, class T, class U, class>
explicit_t<R,T,U> operator/(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, scalar_divide_functor<R,decltype(data(y))>(data(y)));
}

template<class R, class T, class U, class>
explicit_t<R,T,U> operator&&(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, and_functor<R>());
}

template<class G, class T, class U, class>
std::pair<default_t<G,T,U>,default_t<G,T,U>> and_grad(const G& g, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_pair(g, x, y, and_grad_functor<real>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> operator||(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, or_functor<R>());
}

template<class G, class T, class U, class>
std::pair<default_t<G,T,U>,default_t<G,T,U>> or_grad(const G& g, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_pair(g, x, y, or_grad_functor<real>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> operator==(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, equal_functor<R>());
}

template<class G, class T, class U, class>
std::pair<default_t<G,T,U>,default_t<G,T,U>> equal_grad(const G& g, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_pair(g, x, y, equal_grad_functor<real>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> operator!=(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, not_equal_functor<R>());
}

template<class G, class T, class U, class>
std::pair<default_t<G,T,U>,default_t<G,T,U>> not_equal_grad(const G& g,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_pair(g, x, y, not_equal_grad_functor<real>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> operator<(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, less_functor<R>());
}

template<class G, class T, class U, class>
std::pair<default_t<G,T,U>,default_t<G,T,U>> less_grad(const G& g,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_pair(g, x, y, less_grad_functor<real>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> operator<=(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, less_or_equal_functor<R>());
}

template<class G, class T, class U, class>
std::pair<default_t<G,T,U>,default_t<G,T,U>> less_or_equal_grad(const G& g,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_pair(g, x, y, less_or_equal_grad_functor<real>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> operator>(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, greater_functor<R>());
}

template<class G, class T, class U, class>
std::pair<default_t<G,T,U>,default_t<G,T,U>> greater_grad(const G& g,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_pair(g, x, y, greater_grad_functor<real>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> operator>=(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, greater_or_equal_functor<R>());
}

template<class G, class T, class U, class>
std::pair<default_t<G,T,U>,default_t<G,T,U>> greater_or_equal_grad(
    const G& g, const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_pair(g, x, y, greater_or_equal_grad_functor<real>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> copysign(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, copysign_functor<R>());
}

template<class G, class T, class U, class>
std::pair<default_t<G,T,U>,default_t<G,T,U>> copysign_grad(const G& g,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_pair(g, x, y, copysign_grad_functor<real>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> digamma(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, digamma_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> gamma_p(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, gamma_p_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> gamma_q(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, gamma_q_functor<R>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> hadamard(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, hadamard_functor<R>());
}

template<class G, class T, class U, class>
std::pair<default_t<G,T,U>,default_t<G,T,U>> hadamard_grad(const G& g,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_pair(g, x, y, hadamard_grad_functor<real>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> lbeta(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, lbeta_functor<R>());
}

template<class G, class T, class U, class>
std::pair<default_t<G,T,U>,default_t<G,T,U>> lbeta_grad(const G& g,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_pair(g, x, y, lbeta_grad_functor<real>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> lchoose(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, lchoose_functor<R>());
}

template<class G, class T, class U, class>
std::pair<default_t<G,T,U>,default_t<G,T,U>> lchoose_grad(const G& g, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_pair(g, x, y, lchoose_grad_functor<real>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> lgamma(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, lgamma_functor<R>());
}

template<class G, class T, class U, class>
std::pair<default_t<G,T,U>,default_t<G,T,U>> lgamma_grad(const G& g, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_pair(g, x, y, lgamma_grad_functor<real>());
}

template<class R, class T, class U, class>
explicit_t<R,T,U> pow(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, pow_functor<R>());
}

template<class G, class T, class U, class>
std::pair<default_t<G,T,U>,default_t<G,T,U>> pow_grad(const G& g, const T& x,
    const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_pair(g, x, y, pow_grad_functor<real>());
}

}
