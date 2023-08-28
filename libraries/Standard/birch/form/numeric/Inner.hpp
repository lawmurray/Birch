  /**
   * @file
   */
  #pragma once

  #include "birch/form/Form.hpp"

  namespace birch {

template<argument T, argument U = Empty>
struct Inner : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    if constexpr (empty<U>) {
      return numbirch::inner(birch::eval(this->x));
    } else {
      return numbirch::inner(birch::eval(this->x), birch::eval(this->y));
    }
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if constexpr (empty<U>) {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::inner_grad(std::forward<G>(g),
            birch::eval(this->x)), visitor);
      }
    } else {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::inner_grad1(g,
            birch::eval(this->x), birch::eval(this->y)), visitor);
      }
      if (!birch::is_constant(this->y)) {
        birch::shallow_grad(this->y, numbirch::inner_grad2(std::forward<G>(g),
            birch::eval(this->x), birch::eval(this->y)), visitor);
      }
    }
  }

  int rows() const {
    return birch::columns(this->x);
  }

  int columns() const {
    if constexpr (empty<U>) {
      return birch::columns(this->x);
    } else {
      return birch::columns(this->y);
    }
  }
};

template<argument T, argument U>
struct is_form<Inner<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<Inner<T,U>> {
  using type = Inner<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<Inner<T,U>> {
  using type = Inner<peg_t<T>,peg_t<U>>;
};

template<argument T>
auto inner(T&& x) {
  return Inner<tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<argument T, argument U>
auto inner(T&& x, U&& y) {
  return Inner<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

}

#include "birch/form/array/Diagonal.hpp"

namespace birch {

template<argument T, argument U>
auto inner(const Diagonal<T,int>& x, U&& y) {
  return x.x*std::forward<U>(y);
}

template<argument T, argument U>
auto inner(T&& x, const Diagonal<U,int>& y) {
  return transpose(std::forward<T>(x))*y.x;
}

template<argument T, argument U>
auto inner(const Diagonal<T,int>& x, const Diagonal<U,int>& y) {
  assert(x.y == y.y);
  return diagonal(x.x*y.x, x.y);
}

}
