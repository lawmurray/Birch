/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U>
struct Pack : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::pack(birch::eval(this->x), birch::eval(this->y));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::pack_grad1(g,
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
    if (!birch::is_constant(this->y)) {
      birch::shallow_grad(this->y, numbirch::pack_grad2(std::forward<G>(g),
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
  }

  int rows() const {
    return birch::rows(this->x, this->y);
  }

  int columns() const {
    return birch::columns(this->x) + birch::columns(this->y);
  }
};

template<argument T, argument U>
struct is_form<Pack<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<Pack<T,U>> {
  using type = Pack<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<Pack<T,U>> {
  using type = Pack<peg_t<T>,peg_t<U>>;
};

template<argument T, argument U>
auto pack(T&& x, U&& y) {
  return Pack<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

}
