/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct IsNan : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::isnan(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::isnan_grad(std::forward<G>(g),
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

template<argument T>
struct is_form<IsNan<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<IsNan<T>> {
  using type = IsNan<tag_t<T>>;
};

template<argument T>
struct peg_s<IsNan<T>> {
  using type = IsNan<peg_t<T>>;
};

template<argument T>
auto isnan(T&& x) {
  return IsNan<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}
