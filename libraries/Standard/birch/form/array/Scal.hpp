/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Scal : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::scal(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::scal_grad(std::forward<G>(g),
          birch::eval(this->x)), visitor);
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
struct is_form<Scal<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Scal<T>> {
  using type = Scal<tag_t<T>>;
};

template<argument T>
struct peg_s<Scal<T>> {
  using type = Scal<peg_t<T>>;
};

template<argument T>
auto scal(T&& x) {
  return Scal<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}
