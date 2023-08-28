/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U = Empty>
struct Digamma : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    if constexpr (empty<U>) {
      return numbirch::digamma(birch::eval(this->x));
    } else {
      return numbirch::digamma(birch::eval(this->x), birch::eval(this->y));
    }
  }

  int rows() const {
    if constexpr (empty<U>) {
      return birch::rows(this->x);
    } else {
      return birch::rows(this->x, this->y);
    }
  }

  int columns() const {
    if constexpr (empty<U>) {
      return birch::columns(this->x);
    } else {
      return birch::columns(this->x, this->y);
    }
  }
};

template<argument T, argument U>
struct is_form<Digamma<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<Digamma<T,U>> {
  using type = Digamma<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<Digamma<T,U>> {
  using type = Digamma<peg_t<T>,peg_t<U>>;
};

template<argument T>
auto digamma(T&& x) {
  return Digamma<tag_t<T>>{{tag(std::forward<T>(x))}};
}

template<argument T, argument U>
auto digamma(T&& x, U&& y) {
  return Digamma<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

}
