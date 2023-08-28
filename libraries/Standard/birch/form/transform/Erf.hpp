/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Erf : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::erf(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::erf_grad(std::forward<G>(g),
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
struct is_form<Erf<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Erf<T>> {
  using type = Erf<tag_t<T>>;
};

template<argument T>
struct peg_s<Erf<T>> {
  using type = Erf<peg_t<T>>;
};

template<argument T>
auto erf(T&& x) {
  return Erf<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}
