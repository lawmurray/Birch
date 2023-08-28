/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U>
struct Mul : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::mul(birch::eval(this->x), birch::eval(this->y));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::mul_grad1(g,
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
    if (!birch::is_constant(this->y)) {
      birch::shallow_grad(this->y, numbirch::mul_grad2(std::forward<G>(g),
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
  }

  int rows() const {
    if constexpr (numbirch::scalar<decltype(birch::eval(this->x))> ||
        numbirch::scalar<decltype(birch::eval(this->y))>) {
      return birch::rows(this->x, this->y);
    } else {
      return birch::rows(this->x);
    }
  }

  int columns() const {
    if constexpr (numbirch::scalar<decltype(birch::eval(this->x))> ||
        numbirch::scalar<decltype(birch::eval(this->y))>) {
      return birch::columns(this->x, this->y);
    } else {
      return birch::columns(this->y);
    }
  }
};

template<argument T, argument U>
struct is_form<Mul<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<Mul<T,U>> {
  using type = Mul<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<Mul<T,U>> {
  using type = Mul<peg_t<T>,peg_t<U>>;
};

template<argument T, argument U>
auto mul(T&& x, U&& y) {
  return Mul<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

template<argument T, argument U>
requires (!numbirch::arithmetic<T> || !numbirch::arithmetic<U>)
auto operator*(T&& x, U&& y) {
  return mul(std::forward<T>(x), std::forward<U>(y));
}

}

#include "birch/form/array/Diagonal.hpp"

namespace birch {

template<argument T, argument U>
auto operator*(const Diagonal<T,int>& x, U&& y) {
  return diagonal(x.x*y, x.y);
}

template<argument T, argument U>
auto operator*(T&& x, const Diagonal<U,int>& y) {
  return diagonal(x*y.x, y.y);
}

template<argument T, argument U>
auto operator*(const Diagonal<T,int>& x, const Diagonal<U,int>& y) {
  assert(x.y == y.y);
  return diagonal(x.x*y.x, x.y);
}

}
