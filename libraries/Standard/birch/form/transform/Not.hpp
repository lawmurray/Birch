/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Not : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::logical_not(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::logical_not_grad(std::forward<G>(g),
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
struct is_form<Not<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Not<T>> {
  using type = Not<tag_t<T>>;
};

template<argument T>
struct peg_s<Not<T>> {
  using type = Not<peg_t<T>>;
};

template<argument T>
auto logical_not(T&& x) {
  return Not<tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<argument T>
auto logical_not(const Not<T>& x) {
  return x.x;
}

template<argument T>
requires (!numbirch::arithmetic<T>)
auto operator!(T&& x) {
  return logical_not(std::forward<T>(x));
}

}
