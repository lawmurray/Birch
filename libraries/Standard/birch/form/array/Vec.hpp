/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Vec : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::vec(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::vec_grad(std::forward<G>(g),
          birch::eval(this->x)), visitor);
    }
  }

  int rows() const {
    return birch::size(this->x);
  }

  static constexpr int columns() {
    return 1;
  }
};

template<argument T>
struct is_form<Vec<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Vec<T>> {
  using type = Vec<tag_t<T>>;
};

template<argument T>
struct peg_s<Vec<T>> {
  using type = Vec<peg_t<T>>;
};

template<argument T>
auto vec(T&& x) {
  return Vec<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}
