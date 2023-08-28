/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U = Empty>
struct Diagonal : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    if constexpr (empty<U>) {
      return numbirch::diagonal(birch::eval(this->x));
    } else {
      return numbirch::diagonal(birch::eval(this->x), this->y);
    }
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if constexpr (empty<U>) {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::diagonal_grad(std::forward<G>(g),
            birch::eval(this->x)), visitor);
      }
    } else {
      if (!birch::is_constant(this->x)) {
        birch::shallow_grad(this->x, numbirch::diagonal_grad(std::forward<G>(g),
            birch::eval(this->x), this->y), visitor);
      }
    }
  }

  int rows() const {
    if constexpr (empty<U>) {
      return rows(this->x);
    } else {
      return this->y;
    }
  }

  int columns() const {
    if constexpr (empty<U>) {
      return columns(this->x);
    } else {
      return this->y;
    }
  }
};

template<argument T, argument U>
struct is_form<Diagonal<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<Diagonal<T,U>> {
  using type = Diagonal<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<Diagonal<T,U>> {
  using type = Diagonal<peg_t<T>,peg_t<U>>;
};

template<argument T>
auto diagonal(T&& x) {
  return Diagonal<tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<argument T>
auto diagonal(T&& x, const int n) {
  return Diagonal<tag_t<T>,int>{{tag(std::forward<T>(x)), n}};
}

inline auto identity(const int n) {
  return Diagonal<Real,int>{{1.0, n}};
}

}
