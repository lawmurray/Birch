/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Min : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::min(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::min_grad(std::forward<G>(g),
          eval(), birch::eval(this->x)), visitor);
    }
  }

  static constexpr int rows() {
    return 1;
  }

  static constexpr int columns() {
    return 1;
  }
};

template<argument T>
struct is_form<Min<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Min<T>> {
  using type = Min<tag_t<T>>;
};

template<argument T>
struct peg_s<Min<T>> {
  using type = Min<peg_t<T>>;
};

template<argument T>
auto min(T&& x) {
  return Min<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}
