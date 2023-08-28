/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U = Empty>
struct StandardGaussian : public Form<T,U> {
  BIRCH_FORM
  
  auto eval() const {
    if constexpr (empty<U>) {
      return numbirch::standard_gaussian(this->x);
    } else {
      return numbirch::standard_gaussian(this->x, this->y);
    }
  }

  int rows() const {
    return this->x;
  }

  int columns() const {
    if constexpr (empty<U>) {
      return 1;
    } else {
      return this->y;
    }
  }
};

template<argument T, argument U>
struct is_form<StandardGaussian<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<StandardGaussian<T,U>> {
  using type = StandardGaussian<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<StandardGaussian<T,U>> {
  using type = StandardGaussian<peg_t<T>,peg_t<U>>;
};

inline auto standard_gaussian(const int n) {
  return StandardGaussian<int>{{n}};
}

inline auto standard_gaussian(const int m, const int n) {
  return StandardGaussian<int,int>{{m, n}};
}

}
