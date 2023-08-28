/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U>
struct TriSolve : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::trisolve(birch::eval(this->x), birch::eval(this->y));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::trisolve_grad1(std::forward<G>(g),
          eval(), birch::eval(this->x), birch::eval(this->y)), visitor);
    }
    if (!birch::is_constant(this->y)) {
      birch::shallow_grad(this->y, numbirch::trisolve_grad2(std::forward<G>(g),
          eval(), birch::eval(this->x), birch::eval(this->y)), visitor);
    }
  }

  int rows() const {
    if constexpr (numbirch::scalar<decltype(birch::eval(this->y))>) {
      return birch::rows(this->x);
    } else {
      return birch::rows(this->y);
    }
  }

  int columns() const {
    if constexpr (numbirch::scalar<decltype(birch::eval(this->y))>) {
      return birch::columns(this->x);
    } else {
      return birch::columns(this->y);
    }
  }
};

template<argument T, argument U>
struct is_form<TriSolve<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<TriSolve<T,U>> {
  using type = TriSolve<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<TriSolve<T,U>> {
  using type = TriSolve<peg_t<T>,peg_t<U>>;
};

template<argument T, argument U>
auto trisolve(T&& x, U&& y) {
  return TriSolve<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

}

#include "birch/form/array/Diagonal.hpp"

namespace birch {

template<argument T, argument U>
auto trisolve(const Diagonal<T,int>& x, U&& y) {
  return y/x.x;
}

template<argument T, argument U>
auto trisolve(const Diagonal<T>& x, U&& y) {
  return y/x.x;
}

}
