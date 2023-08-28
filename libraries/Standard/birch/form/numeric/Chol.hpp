/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Chol : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::chol(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::chol_grad(std::forward<G>(g),
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
struct is_form<Chol<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Chol<T>> {
  using type = Chol<tag_t<T>>;
};

template<argument T>
struct peg_s<Chol<T>> {
  using type = Chol<peg_t<T>>;
};

template<argument T>
auto chol(T&& x) {
  return Chol<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}

#include "birch/form/array/Diagonal.hpp"
#include "birch/form/transform/Sqrt.hpp"

namespace birch {

template<argument T>
auto chol(const Diagonal<T,int>& x) {
  return diagonal(sqrt(x.x), x.y);
}

template<argument T>
auto chol(const Diagonal<T>& x) {
  return diagonal(sqrt(x.x));
}

}
