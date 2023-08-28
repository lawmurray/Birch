/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U, argument V>
struct Where : public Form<T,U,V> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::where(birch::eval(this->x), birch::eval(this->y),
        birch::eval(this->z));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::where_grad1(g,
          birch::eval(this->x), birch::eval(this->y), birch::eval(this->z)),
          visitor);
    }
    if (!birch::is_constant(this->y)) {
      birch::shallow_grad(this->y, numbirch::where_grad2(g,
          birch::eval(this->x), birch::eval(this->y), birch::eval(this->z)),
          visitor);
    }
    if (!birch::is_constant(this->z)) {
      birch::shallow_grad(this->z, numbirch::where_grad3(std::forward<G>(g),
          birch::eval(this->x), birch::eval(this->y), birch::eval(this->z)),
          visitor);
    }
  }

  int rows() const {
    return birch::rows(this->x, this->y, this->z);
  }

  int columns() const {
    return birch::columns(this->x, this->y, this->z);
  }
};

template<argument T, argument U, argument V>
struct is_form<Where<T,U,V>> {
  static constexpr bool value = true;
};

template<argument T, argument U, argument V>
struct tag_s<Where<T,U,V>> {
  using type = Where<tag_t<T>,tag_t<U>,tag_t<V>>;
};

template<argument T, argument U, argument V>
struct peg_s<Where<T,U,V>> {
  using type = Where<peg_t<T>,peg_t<U>,peg_t<V>>;
};

template<argument T, argument U, argument V>
auto where(T&& x, U&& y, V&& z) {
  return Where<tag_t<T>,tag_t<U>,tag_t<V>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y)), tag(std::forward<V>(z))}};
}

}
