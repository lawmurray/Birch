/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Exp : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::exp(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::exp_grad(std::forward<G>(g),
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
struct is_form<Exp<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Exp<T>> {
  using type = Exp<tag_t<T>>;
};

template<argument T>
struct peg_s<Exp<T>> {
  using type = Exp<peg_t<T>>;
};

template<argument T>
auto exp(T&& x) {
  return Exp<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}

#include "birch/form/transform/Log.hpp"

namespace birch {

template<argument T>
decltype(auto) exp(const Log<T>& x) {
  return x.x;
}

}
