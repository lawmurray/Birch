/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct LDet : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::ldet(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::ldet_grad(std::forward<G>(g),
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
struct is_form<LDet<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<LDet<T>> {
  using type = LDet<tag_t<T>>;
};

template<argument T>
struct peg_s<LDet<T>> {
  using type = LDet<peg_t<T>>;
};

template<argument T>
auto ldet(T&& x) {
  return LDet<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}

#include "birch/form/array/Diagonal.hpp"
#include "birch/form/numeric/Mul.hpp"
#include "birch/form/transform/Log.hpp"
#include "birch/form/reduce/Sum.hpp"

namespace birch {

template<argument T>
auto ldet(const Diagonal<T,int>& x) {
  return x.y*log(x.x);
}

template<argument T>
auto ldet(const Diagonal<T>& x) {
  return sum(log(x.x));
}

}
