/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Iota : public Form<T,int> {
  BIRCH_FORM
  
  auto eval() const {
    return numbirch::iota(birch::eval(this->x), this->y);
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::iota_grad(std::forward<G>(g),
          birch::eval(this->x), this->y), visitor);
    }
  }

  int rows() const {
    return this->y;
  }

  int columns() const {
    return 1;
  }
};

template<argument T>
struct is_form<Iota<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Iota<T>> {
  using type = Iota<tag_t<T>>;
};

template<argument T>
struct peg_s<Iota<T>> {
  using type = Iota<peg_t<T>>;
};

template<argument T>
auto iota(T&& x, const int n) {
  return Iota<tag_t<T>>{{tag(std::forward<T>(x)), n}};
}

}
