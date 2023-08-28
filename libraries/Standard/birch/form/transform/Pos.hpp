/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Pos : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::pos(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::pos_grad(std::forward<G>(g),
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
struct is_form<Pos<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Pos<T>> {
  using type = Pos<tag_t<T>>;
};

template<argument T>
struct peg_s<Pos<T>> {
  using type = Pos<peg_t<T>>;
};

template<argument T>
auto pos(T&& x) {
  return Pos<tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<argument T>
auto pos(const Pos<T>& x) {
  return x;
}

template<argument T>
requires (!numbirch::arithmetic<T>)
auto operator+(T&& x) {
  return pos(std::forward<T>(x));
}

}
