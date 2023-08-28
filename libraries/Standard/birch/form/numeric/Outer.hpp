/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U = Empty>
struct Outer : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    if constexpr (empty<U>) {
      return numbirch::outer(birch::eval(this->x));
    } else {
      return numbirch::outer(birch::eval(this->x), birch::eval(this->y));
    }
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if constexpr (empty<U>) {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::outer_grad(std::forward<G>(g),
            birch::eval(this->x)), visitor);
      }
    } else {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::outer_grad1(g,
            birch::eval(this->x), birch::eval(this->y)), visitor);
      }
      if (!birch::is_constant(this->y)) {
        birch::shallow_grad(this->y, numbirch::outer_grad2(std::forward<G>(g),
            birch::eval(this->x), birch::eval(this->y)), visitor);
      }
    }
  }

  int rows() const {
    return birch::rows(this->x);
  }

  int columns() const {
    if constexpr (empty<U>) {
      return birch::rows(this->x);
    } else {
      return birch::rows(this->y);
    }
  }
};

template<argument T, argument U>
struct is_form<Outer<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<Outer<T,U>> {
  using type = Outer<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<Outer<T,U>> {
  using type = Outer<peg_t<T>,peg_t<U>>;
};

template<argument T>
auto outer(T&& x) {
  return Outer<tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<argument T, argument U>
auto outer(T&& x, U&& y) {
  return Outer<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
    tag(std::forward<U>(y))}};
}

}
