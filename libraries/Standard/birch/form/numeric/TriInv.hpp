/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct TriInv : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::triinv(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::triinv_grad(std::forward<G>(g),
          eval(), birch::eval(this->x)), visitor);
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
struct is_form<TriInv<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<TriInv<T>> {
  using type = TriInv<tag_t<T>>;
};

template<argument T>
struct peg_s<TriInv<T>> {
  using type = TriInv<peg_t<T>>;
};

template<argument T>
auto triinv(T&& x) {
  return TriInv<tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<argument T>
auto triinv(const TriInv<T>& x) {
  return x.x;
}

}

#include "birch/form/array/Diagonal.hpp"
#include "birch/form/transform/Div.hpp"

namespace birch {

template<argument T>
auto triinv(const Diagonal<T,int>& x) {
  return diagonal(1.0/x.x, x.y);
}

template<argument T>
auto triinv(const Diagonal<T>& x) {
  return diagonal(1.0/x.x);
}

}
