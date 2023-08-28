/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U = Empty>
struct TriOuter : public Form<T,U> {
  BIRCH_FORM
  
  auto eval() const {
    if constexpr (empty<U>) {
      return numbirch::triouter(birch::eval(this->x));
    } else {
      return numbirch::triouter(birch::eval(this->x), birch::eval(this->y));
    }
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if constexpr (empty<U>) {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::triouter_grad(std::forward<G>(g),
            birch::eval(this->x)), visitor);
      }
    } else {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::triouter_grad1(g,
            birch::eval(this->x), birch::eval(this->y)), visitor);
      }
      if (!birch::is_constant(this->y)) {
        birch::shallow_grad(this->y, numbirch::triouter_grad2(std::forward<G>(g),
            birch::eval(this->x), birch::eval(this->y)), visitor);
      }
    }
  }

  int rows() const {
    return birch::rows(this->x);
  }

  int columns() const {
    if constexpr (empty<U>) {
      return birch::rows(this->x);
    } else {
      return birch::rows(this->y);
    }
  }
};

template<argument T, argument U>
struct is_form<TriOuter<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<TriOuter<T,U>> {
  using type = TriOuter<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<TriOuter<T,U>> {
  using type = TriOuter<peg_t<T>,peg_t<U>>;
};

template<argument T>
auto triouter(T&& x) {
  return TriOuter<tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<argument T, argument U>
auto triouter(T&& x, U&& y) {
  return TriOuter<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
    tag(std::forward<U>(y))}};
}

}

#include "birch/form/array/Diagonal.hpp"

namespace birch {

template<argument T, argument U>
auto triouter(const Diagonal<T,int>& x, U&& y) {
  return x.x*std::forward<U>(y);
}

template<argument T, argument U>
auto triouter(T&& x, const Diagonal<U,int>& y) {
  return std::forward<T>(x)*y.x;
}

template<argument T, argument U>
auto triouter(const Diagonal<T,int>& x, const Diagonal<U,int>& y) {
  assert(x.y == y.y);
  return diagonal(x.x*y.x, x.y);
}

}
