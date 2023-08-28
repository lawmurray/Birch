/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct CholInv : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::cholinv(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::cholinv_grad(std::forward<G>(g),
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
struct is_form<CholInv<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<CholInv<T>> {
  using type = CholInv<tag_t<T>>;
};

template<argument T>
struct peg_s<CholInv<T>> {
  using type = CholInv<peg_t<T>>;
};

template<argument T>
auto cholinv(T&& x) {
  return CholInv<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}

#include "birch/form/array/Diagonal.hpp"
#include "birch/form/transform/Pow.hpp"

namespace birch {

template<argument T>
auto cholinv(const Diagonal<T,int>& x) {
  return diagonal(pow(x.x, -2.0), x.y);
}

template<argument T>
auto cholinv(const Diagonal<T>& x) {
  return diagonal(pow(x.x, -2.0));
}

}
