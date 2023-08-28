/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U>
struct Pow : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::pow(birch::eval(this->x), birch::eval(this->y));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::pow_grad1(g,
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
    if (!birch::is_constant(this->y)) {
      birch::shallow_grad(this->y, numbirch::pow_grad2(std::forward<G>(g),
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
struct is_form<Pow<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<Pow<T,U>> {
  using type = Pow<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<Pow<T,U>> {
  using type = Pow<peg_t<T>,peg_t<U>>;
};

template<argument T, argument U>
auto pow(T&& x, U&& y) {
  return Pow<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

}
