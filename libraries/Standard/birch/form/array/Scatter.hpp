/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U, argument V, argument W = Empty,
    argument X = Empty>
struct Scatter : public Form<T,U,V,W,X> {
  BIRCH_FORM
  
  auto eval() const {
    if constexpr (empty<W>) {
      return numbirch::scatter(birch::eval(this->x), birch::eval(this->y),
          this->z);
    } else {
      return numbirch::scatter(birch::eval(this->x), birch::eval(this->y),
          birch::eval(this->z), this->a, this->b);
    }
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if constexpr (empty<W>) {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::scatter_grad1(g,
            birch::eval(this->x), birch::eval(this->y), this->z), visitor);
      }
      if (!birch::is_constant(this->y)) {
        birch::shallow_grad(this->y, numbirch::scatter_grad2(std::forward<G>(g),
            birch::eval(this->x), birch::eval(this->y), this->z), visitor);
      }
    } else {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::scatter_grad1(g,
            birch::eval(this->x), birch::eval(this->y), birch::eval(this->z),
            this->a, this->b), visitor);
      }
      if (!birch::is_constant(this->y)) {
        birch::shallow_grad(this->y, numbirch::scatter_grad2(std::forward<G>(g),
            birch::eval(this->x), birch::eval(this->y), birch::eval(this->z),
            this->a, this->b), visitor);
      }
      if (!birch::is_constant(this->z)) {
        birch::shallow_grad(this->z, numbirch::scatter_grad3(std::forward<G>(g),
            birch::eval(this->x), birch::eval(this->y), birch::eval(this->z),
            this->a, this->b), visitor);
      }
    }
  }

  int rows() const {
    if constexpr (empty<W>) {
      return this->z;
    } else {
      return this->a;
    }
  }

  int columns() const {
    if constexpr (empty<W>) {
      return 1;
    } else {
      return this->b;
    }
  }
};

template<argument T, argument U, argument V, argument W, argument X>
struct is_form<Scatter<T,U,V,W,X>> {
  static constexpr bool value = true;
};

template<argument T, argument U, argument V, argument W, argument X>
struct tag_s<Scatter<T,U,V,W,X>> {
  using type = Scatter<tag_t<T>,tag_t<U>,tag_t<V>,tag_t<W>,tag_t<X>>;
};

template<argument T, argument U, argument V, argument W, argument X>
struct peg_s<Scatter<T,U,V,W,X>> {
  using type = Scatter<peg_t<T>,peg_t<U>,peg_t<V>,peg_t<W>,peg_t<X>>;
};

template<argument T, argument U>
auto scatter(T&& x, U&& y, const int n) {
  return Scatter<tag_t<T>,tag_t<U>,int>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y)), n}};
}

template<argument T, argument U, argument V>
auto scatter(T&& x, U&& y, V&& z, const int m, const int n) {
  return Scatter<tag_t<T>,tag_t<U>,tag_t<V>,int,int>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y)), tag(std::forward<V>(z)), m, n}};
}

}
