/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U>
struct TriInnerSolve : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::triinnersolve(birch::eval(this->x), birch::eval(this->y));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::triinnersolve_grad1(std::forward<G>(g),
          eval(), birch::eval(this->x), birch::eval(this->y)), visitor);
    }
    if (!birch::is_constant(this->y)) {
      birch::shallow_grad(this->y, numbirch::triinnersolve_grad2(std::forward<G>(g),
          eval(), birch::eval(this->x), birch::eval(this->y)), visitor);
    }
  }

  int rows() const {
    if constexpr (numbirch::scalar<decltype(birch::eval(this->y))>) {
      return birch::rows(this->x);
    } else {
      return birch::rows(this->y);
    }
  }

  int columns() const {
    if constexpr (numbirch::scalar<decltype(birch::eval(this->y))>) {
      return birch::columns(this->x);
    } else {
      return birch::columns(this->y);
    }
  }
};

template<argument T, argument U>
struct is_form<TriInnerSolve<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<TriInnerSolve<T,U>> {
  using type = TriInnerSolve<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<TriInnerSolve<T,U>> {
  using type = TriInnerSolve<peg_t<T>,peg_t<U>>;
};

template<argument T, argument U>
auto triinnersolve(T&& x, U&& y) {
  return TriInnerSolve<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

}
