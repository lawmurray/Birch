/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Log1p : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::log1p(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::log1p_grad(std::forward<G>(g),
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
struct is_form<Log1p<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Log1p<T>> {
  using type = Log1p<tag_t<T>>;
};

template<argument T>
struct peg_s<Log1p<T>> {
  using type = Log1p<peg_t<T>>;
};

template<argument T>
auto log1p(T&& x) {
  return Log1p<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}

#include "birch/form/transform/Expm1.hpp"

namespace birch {

template<argument T>
decltype(auto) log1p(const Expm1<T>& x) {
  return x.x;
}

}
