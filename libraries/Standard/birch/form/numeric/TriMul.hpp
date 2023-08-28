/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U>
struct TriMul : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::trimul(birch::eval(this->x), birch::eval(this->y));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::trimul_grad1(g,
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
    if (!birch::is_constant(this->y)) {
      birch::shallow_grad(this->y, numbirch::trimul_grad2(std::forward<G>(g),
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
  }

  int rows() const {
    if constexpr (numbirch::scalar<decltype(birch::eval(this->x))> ||
        numbirch::scalar<decltype(birch::eval(this->y))>) {
      return birch::rows(this->x, this->y);
    } else {
      return birch::rows(this->x);
    }
  }

  int columns() const {
    if constexpr (numbirch::scalar<decltype(birch::eval(this->x))> ||
        numbirch::scalar<decltype(birch::eval(this->y))>) {
      return birch::columns(this->x, this->y);
    } else {
      return birch::columns(this->y);
    }
  }
};

template<argument T, argument U>
struct is_form<TriMul<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<TriMul<T,U>> {
  using type = TriMul<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<TriMul<T,U>> {
  using type = TriMul<peg_t<T>,peg_t<U>>;
};

template<argument T, argument U>
auto trimul(T&& x, U&& y) {
  return TriMul<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

}

#include "birch/form/array/Diagonal.hpp"

namespace birch {

template<argument T, argument U>
auto trimul(const Diagonal<T,int>& x, U&& y) {
  return diagonal(x.x*y, x.y);
}

template<argument T, argument U>
auto trimul(T&& x, const Diagonal<U,int>& y) {
  return std::forward<T>(x)*y.x;
}

template<argument T, argument U>
auto trimul(const Diagonal<T,int>& x, const Diagonal<U,int>& y) {
  assert(x.y == y.y);
  return diagonal(x.x*y.x, x.y);
}

}
