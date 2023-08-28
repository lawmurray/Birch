/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct LCholDet : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::lcholdet(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::lcholdet_grad(std::forward<G>(g),
          birch::eval(this->x)), visitor);
    }
  }

  static constexpr int rows() {
    return 1;
  }

  static constexpr int columns() {
    return 1;
  }
};

template<argument T>
struct is_form<LCholDet<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<LCholDet<T>> {
  using type = LCholDet<tag_t<T>>;
};

template<argument T>
struct peg_s<LCholDet<T>> {
  using type = LCholDet<peg_t<T>>;
};

template<argument T>
auto lcholdet(T&& x) {
  return LCholDet<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}

#include "birch/form/array/Diagonal.hpp"
#include "birch/form/numeric/Mul.hpp"
#include "birch/form/transform/Log.hpp"
#include "birch/form/reduce/Sum.hpp"

namespace birch {

template<argument T>
auto lcholdet(const Diagonal<T,int>& x) {
  return 2.0*x.y*log(x.x);
}

template<argument T>
auto lcholdet(const Diagonal<T>& x) {
  return 2.0*sum(log(x.x));
}

}
