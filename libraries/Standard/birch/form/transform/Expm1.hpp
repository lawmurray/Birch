/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Expm1 : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::expm1(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::expm1_grad(std::forward<G>(g),
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
struct is_form<Expm1<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Expm1<T>> {
  using type = Expm1<tag_t<T>>;
};

template<argument T>
struct peg_s<Expm1<T>> {
  using type = Expm1<peg_t<T>>;
};

template<argument T>
auto expm1(T&& x) {
  return Expm1<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}

#include "birch/form/transform/Log1p.hpp"

namespace birch {

template<argument T>
decltype(auto) expm1(const Log1p<T>& x) {
  return x.x;
}

}
