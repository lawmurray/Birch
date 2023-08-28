/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Transpose : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::transpose(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::transpose_grad(std::forward<G>(g),
          birch::eval(this->x)), visitor);
    }
  }

  int rows() const {
    return birch::columns(this->x);
  }

  int columns() const {
    return birch::rows(this->x);
  }
};

template<argument T>
struct is_form<Transpose<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Transpose<T>> {
  using type = Transpose<tag_t<T>>;
};

template<argument T>
struct peg_s<Transpose<T>> {
  using type = Transpose<peg_t<T>>;
};

template<argument T>
auto transpose(T&& x) {
  return Transpose<tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<argument T>
decltype(auto) transpose(const Transpose<T>& x) {
  return x.x;
}

}
