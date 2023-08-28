/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Floor : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::floor(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::floor_grad(std::forward<G>(g),
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
struct is_form<Floor<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Floor<T>> {
  using type = Floor<tag_t<T>>;
};

template<argument T>
struct peg_s<Floor<T>> {
  using type = Floor<peg_t<T>>;
};

template<argument T>
auto floor(T&& x) {
  return Floor<tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<argument T>
auto floor(const Floor<T>& x) {
  return x;
}

}
