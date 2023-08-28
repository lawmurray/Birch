/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U, argument V>
struct IBeta : public Form<T,U,V> {
  BIRCH_FORM
  
  auto eval() const {
    return numbirch::ibeta(birch::eval(this->x), birch::eval(this->y),
        birch::eval(this->z));
  }

  int rows() const {
    return birch::rows(this->x, this->y, this->z);
  }

  int columns() const {
    return birch::columns(this->x, this->y, this->z);
  }
};

template<argument T, argument U, argument V>
struct is_form<IBeta<T,U,V>> {
  static constexpr bool value = true;
};

template<argument T, argument U, argument V>
struct tag_s<IBeta<T,U,V>> {
  using type = IBeta<tag_t<T>,tag_t<U>,tag_t<V>>;
};

template<argument T, argument U, argument V>
struct peg_s<IBeta<T,U,V>> {
  using type = IBeta<peg_t<T>,peg_t<U>,peg_t<V>>;
};

template<argument T, argument U, argument V>
auto ibeta(T&& x, U&& y, V&& z) {
  return IBeta<tag_t<T>,tag_t<U>,tag_t<V>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y)), tag(std::forward<V>(z))}};
}

}
