/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U, argument V = Empty>
struct Element : public Form<T,U,V> {
  BIRCH_FORM

  auto eval() const {
    if constexpr (empty<V>) {
      return numbirch::element(birch::eval(this->x), birch::eval(this->y));
    } else {
      return numbirch::element(birch::eval(this->x), birch::eval(this->y),
          birch::eval(this->z));
    }
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if constexpr (empty<V>) {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::element_grad1(g,
            birch::eval(this->x), birch::eval(this->y)), visitor);
      }
      if (!birch::is_constant(this->y)) {
        birch::shallow_grad(this->y, numbirch::element_grad2(std::forward<G>(g),
            birch::eval(this->x), birch::eval(this->y)), visitor);
      }
    } else {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::element_grad1(g,
            birch::eval(this->x), birch::eval(this->y), birch::eval(this->z)),
            visitor);
      }
      if (!birch::is_constant(this->y)) {
        birch::shallow_grad(this->y, numbirch::element_grad2(g,
            birch::eval(this->x), birch::eval(this->y), birch::eval(this->z)),
            visitor);
      }
      if (!birch::is_constant(this->z)) {
        birch::shallow_grad(this->z, numbirch::element_grad3(g,
            birch::eval(this->x), birch::eval(this->y), birch::eval(this->z)),
            visitor);
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

template<argument T, argument U, argument V>
struct is_form<Element<T,U,V>> {
  static constexpr bool value = true;
};

template<argument T, argument U, argument V>
struct tag_s<Element<T,U,V>> {
  using type = Element<tag_t<T>,tag_t<U>,tag_t<V>>;
};

template<argument T, argument U, argument V>
struct peg_s<Element<T,U,V>> {
  using type = Element<peg_t<T>,peg_t<U>,peg_t<V>>;
};

template<argument T, argument U>
auto element(T&& x, U&& y) {
  return Element<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

template<argument T, argument U, argument V>
auto element(T&& x, U&& y, V&& z) {
  return Element<tag_t<T>,tag_t<U>,tag_t<V>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y)), tag(std::forward<V>(z))}};
}

}
