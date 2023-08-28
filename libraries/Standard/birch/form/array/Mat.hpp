/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Mat : public Form<T,int> {
  BIRCH_FORM
  
  auto eval() const {
    return numbirch::mat(birch::eval(this->x), this->y);
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::mat_grad(std::forward<G>(g),
          birch::eval(this->x), this->y), visitor);
    }
  }

  int rows() const {
    return birch::size(this->x)/this->y;
  }

  int columns() const {
    return this->y;
  }
};

template<argument T>
struct is_form<Mat<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Mat<T>> {
  using type = Mat<tag_t<T>>;
};

template<argument T>
struct peg_s<Mat<T>> {
  using type = Mat<peg_t<T>>;
};

template<argument T>
auto mat(T&& x, const int n) {
  return Mat<tag_t<T>>{{tag(std::forward<T>(x)), n}};
}

}
