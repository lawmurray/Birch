/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<numbirch::arithmetic To, argument T>
struct Cast : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::cast<To>(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::cast_grad<To>(std::forward<G>(g),
          birch::eval(this->x)), visitor);
    }
  }

  int rows() const {
    return birch::rows(this->x);
  }

  int columns() const {
    return birch::columns(this->x);
  }
};

template<numbirch::arithmetic To, argument T>
struct is_form<Cast<To,T>> {
  static constexpr bool value = true;
};

template<numbirch::arithmetic To, argument T>
struct tag_s<Cast<To,T>> {
  using type = Cast<To,tag_t<T>>;
};

template<numbirch::arithmetic To, argument T>
struct peg_s<Cast<To,T>> {
  using type = Cast<To,peg_t<T>>;
};

template<numbirch::arithmetic To, argument T>
auto cast(T&& x) {
  return Cast<To,tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<numbirch::arithmetic To, argument T>
auto cast(const Cast<To,T>& x) {
  return x;
}

}
