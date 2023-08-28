/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U>
struct MatrixStack : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::stack(birch::eval(this->x), birch::eval(this->y));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::stack_grad1(g,
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
    if (!birch::is_constant(this->y)) {
      birch::shallow_grad(this->y, numbirch::stack_grad2(std::forward<G>(g),
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
  }

  int rows() const {
    return birch::rows(this->x) + birch::rows(this->y);
  }

  int columns() const {
    return birch::columns(this->x, this->y);
  }
};

template<argument T, argument U>
struct is_form<MatrixStack<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<MatrixStack<T,U>> {
  using type = MatrixStack<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<MatrixStack<T,U>> {
  using type = MatrixStack<peg_t<T>,peg_t<U>>;
};

template<argument T, argument U>
auto stack(T&& x, U&& y) {
  return MatrixStack<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

}
