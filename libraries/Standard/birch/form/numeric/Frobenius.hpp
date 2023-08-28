/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U = Empty>
struct Frobenius : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    if constexpr (empty<U>) {
      return numbirch::frobenius(birch::eval(this->x));
    } else {
      return numbirch::frobenius(birch::eval(this->x), birch::eval(this->y));
    }
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if constexpr (empty<U>) {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::frobenius_grad(std::forward<G>(g),
            birch::eval(this->x)), visitor);
      }
    } else {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::frobenius_grad1(g,
            birch::eval(this->x), birch::eval(this->y)), visitor);
      }
      if (!birch::is_constant(this->y)) {
        birch::shallow_grad(this->y, numbirch::frobenius_grad2(std::forward<G>(g),
            birch::eval(this->x), birch::eval(this->y)), visitor);
      }
    }
  }

  static constexpr int rows() {
    return 1;
  }

  static constexpr int columns() {
    return 1;
  }
};

template<argument T, argument U>
struct is_form<Frobenius<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<Frobenius<T,U>> {
  using type = Frobenius<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<Frobenius<T,U>> {
  using type = Frobenius<peg_t<T>,peg_t<U>>;
};

template<argument T>
auto frobenius(T&& x) {
  return Frobenius<tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<argument T, argument U>
auto frobenius(T&& x, U&& y) {
  return Frobenius<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

}
