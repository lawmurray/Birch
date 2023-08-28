/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U>
struct Convolve : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::convolve(birch::eval(this->x), birch::eval(this->y));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::convolve_grad1(g,
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
    if (!birch::is_constant(this->y)) {
      birch::shallow_grad(this->y, numbirch::convolve_grad2(std::forward<G>(g),
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
  }

  int rows() const {
    return birch::rows(this->x) + birch::rows(this->y) - 1;
  }

  static constexpr int columns() {
    return 1;
  }
};

template<argument T, argument U>
struct is_form<Convolve<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<Convolve<T,U>> {
  using type = Convolve<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<Convolve<T,U>> {
  using type = Convolve<peg_t<T>,peg_t<U>>;
};

template<argument T, argument U>
auto convolve(T&& x, U&& y) {
  return Convolve<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

}
