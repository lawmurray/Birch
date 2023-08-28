/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Round : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::round(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::round_grad(std::forward<G>(g),
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
struct is_form<Round<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Round<T>> {
  using type = Round<tag_t<T>>;
};

template<argument T>
struct peg_s<Round<T>> {
  using type = Round<peg_t<T>>;
};

template<argument T>
auto round(T&& x) {
  return Round<tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<argument T>
auto round(const Round<T>& x) {
  return x;
}

}
