/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U>
struct And : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::logical_and(birch::eval(this->x), birch::eval(this->y));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::logical_and_grad1(g,
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
    if (!birch::is_constant(this->y)) {
      birch::shallow_grad(this->y, numbirch::logical_and_grad2(std::forward<G>(g),
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
  }

  int rows() const {
    return birch::rows(this->x, this->y);
  }

  int columns() const {
    return birch::columns(this->x, this->y);
  }
};

template<argument T, argument U>
struct is_form<And<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<And<T,U>> {
  using type = And<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<And<T,U>> {
  using type = And<peg_t<T>,peg_t<U>>;
};

template<argument T, argument U>
auto logical_and(T&& x, U&& y) {
  return And<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

/* principle of least surprise: do not overload && as it will not short
 * circuit, leading to unintuitive errors */

}
