/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U, argument V = Empty>
struct Fill : public Form<T,U,V> {
  BIRCH_FORM
  
  auto eval() const {
    if constexpr (empty<V>) {
      return numbirch::fill(birch::eval(this->x), this->y);
    } else {
      return numbirch::fill(birch::eval(this->x), this->y, this->z);
    }
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if constexpr (empty<V>) {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::fill_grad(g,
            birch::eval(this->x), this->y), visitor);
      }
    } else {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::fill_grad(g,
            birch::eval(this->x), this->y, this->z), visitor);
      }
    }
  }

  int rows() const {
    return this->y;
  }

  int columns() const {
    if constexpr (empty<V>) {
      return 1;
    } else {
      return this->z;
    }
  }
};

template<argument T, argument U, argument V>
struct is_form<Fill<T,U,V>> {
  static constexpr bool value = true;
};

template<argument T, argument U, argument V>
struct tag_s<Fill<T,U,V>> {
  using type = Fill<tag_t<T>,tag_t<U>,tag_t<V>>;
};

template<argument T, argument U, argument V>
struct peg_s<Fill<T,U,V>> {
  using type = Fill<peg_t<T>,peg_t<U>,peg_t<V>>;
};

template<argument T>
auto fill(T&& x, const int n) {
  return Fill<tag_t<T>,int>{{tag(std::forward<T>(x)), n}};
}

template<argument T>
auto fill(T&& x, const int m, const int n) {
  return Fill<tag_t<T>,int,int>{{tag(std::forward<T>(x)), m, n}};
}

}
