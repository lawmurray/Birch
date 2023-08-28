/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U, argument V = Empty>
struct Gather : public Form<T,U,V> {
  BIRCH_FORM

  auto eval() const {
    if constexpr (empty<V>) {
      return numbirch::gather(birch::eval(this->x), birch::eval(this->y));
    } else {
      return numbirch::gather(birch::eval(this->x), birch::eval(this->y),
          birch::eval(this->z));
    }
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if constexpr (empty<V>) {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::gather_grad1(g,
            birch::eval(this->x), birch::eval(this->y)), visitor);
      }
      if (!birch::is_constant(this->y)) {
        birch::shallow_grad(this->y, numbirch::gather_grad2(std::forward<G>(g),
            birch::eval(this->x), birch::eval(this->y)), visitor);
      }
    } else {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::gather_grad1(g,
            birch::eval(this->x), birch::eval(this->y), birch::eval(this->z)),
            visitor);
      }
      if (!birch::is_constant(this->y)) {
        birch::shallow_grad(this->y, numbirch::gather_grad2(std::forward<G>(g),
            birch::eval(this->x), birch::eval(this->y), birch::eval(this->z)),
            visitor);
      }
      if (!birch::is_constant(this->z)) {
        birch::shallow_grad(this->z, numbirch::gather_grad3(std::forward<G>(g),
            birch::eval(this->x), birch::eval(this->y), birch::eval(this->z)),
            visitor);
      }
    }
  }

  int rows() const {
    return birch::rows(this->y);
  }

  int columns() const {
    return birch::columns(this->y);
  }
};

template<argument T, argument U, argument V>
struct is_form<Gather<T,U,V>> {
  static constexpr bool value = true;
};

template<argument T, argument U, argument V>
struct tag_s<Gather<T,U,V>> {
  using type = Gather<tag_t<T>,tag_t<U>,tag_t<V>>;
};

template<argument T, argument U, argument V>
struct peg_s<Gather<T,U,V>> {
  using type = Gather<peg_t<T>,peg_t<U>,peg_t<V>>;
};

template<argument T, argument U>
auto gather(T&& x, U&& y) {
  return Gather<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

template<argument T, argument U, argument V>
auto gather(T&& x, U&& y, V&& z) {
  return Gather<tag_t<T>,tag_t<U>,tag_t<V>>{{ tag(std::forward<T>(x)),
      tag(std::forward<U>(y)), tag(std::forward<V>(z))}};
}

}
