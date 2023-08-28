/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Ceil : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::ceil(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::ceil_grad(std::forward<G>(g),
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
struct is_form<Ceil<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Ceil<T>> {
  using type = Ceil<tag_t<T>>;
};

template<argument T>
struct peg_s<Ceil<T>> {
  using type = Ceil<peg_t<T>>;
};

template<argument T>
auto ceil(T&& x) {
  return Ceil<tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<argument T>
auto ceil(const Ceil<T>& x) {
  return x;
}

}
