/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Inv : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::inv(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::inv_grad(std::forward<G>(g),
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
struct is_form<Inv<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Inv<T>> {
  using type = Inv<tag_t<T>>;
};

template<argument T>
struct peg_s<Inv<T>> {
  using type = Inv<peg_t<T>>;
};

template<argument T>
auto inv(T&& x) {
  return Inv<tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<argument T>
auto inv(const Inv<T>& x) {
  return x.x;
}

}

#include "birch/form/array/Diagonal.hpp"
#include "birch/form/transform/Div.hpp"

namespace birch {

template<argument T>
auto inv(const Diagonal<T,int>& x) {
  return diagonal(1.0/x.x, x.y);
}

template<argument T>
auto inv(const Diagonal<T>& x) {
  return diagonal(1.0/x.x);
}

}
