/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U>
struct Add : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::add(birch::eval(this->x), birch::eval(this->y));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::add_grad1(g,
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
    if (!birch::is_constant(this->y)) {
      birch::shallow_grad(this->y, numbirch::add_grad2(std::forward<G>(g),
          birch::eval(this->x), birch::eval(this->y)), visitor);
    }
  }

  int rows() const {
    return birch::rows(this->x, this->y);
  }

  int columns() const {
    return birch::columns(this->x, this->y);
  }
};

template<argument T, argument U>
struct is_form<Add<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<Add<T,U>> {
  using type = Add<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<Add<T,U>> {
  using type = Add<peg_t<T>,peg_t<U>>;
};

template<argument T, argument U>
auto add(T&& x, U&& y) {
  return Add<tag_t<T>,tag_t<U>>{{
      tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

template<argument T, argument U>
requires (!numbirch::arithmetic<T> || !numbirch::arithmetic<U>)
auto operator+(T&& x, U&& y) {
  return add(std::forward<T>(x), std::forward<U>(y));
}

}

#include "birch/form/array/Fill.hpp"

namespace birch {

template<argument T, argument U>
requires (numbirch::arithmetic<decltype(eval(std::declval<U>()))>)
auto operator+(const Fill<T,int>& x, U&& y) {
  return fill(x.x + std::forward<U>(y), x.y);
}

template<argument T, argument U>
requires (numbirch::arithmetic<decltype(eval(std::declval<T>()))>)
auto operator+(T&& x, const Fill<U,int>& y) {
  return fill(std::forward<T>(x) + y.x, y.y);
}

template<argument T, argument U>
requires (numbirch::arithmetic<decltype(eval(std::declval<U>()))>)
auto operator+(const Fill<T,int,int>& x, U&& y) {
  return fill(x.x + std::forward<U>(y), x.y, x.z);
}

template<argument T, argument U>
requires (numbirch::arithmetic<decltype(eval(std::declval<T>()))>)
auto operator+(T&& x, const Fill<U,int,int>& y) {
  return fill(std::forward<T>(x) + y.x, y.y, y.z);
}

}
