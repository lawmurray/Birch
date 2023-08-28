/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct SimulateWishart : public Form<T,int> {
  BIRCH_FORM
  
  auto eval() const {
    return numbirch::simulate_wishart(birch::eval(this->x), this->y);
  }

  int rows() const {
    return this->y;
  }

  int columns() const {
    return this->y;
  }
};

template<argument T>
struct is_form<SimulateWishart<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<SimulateWishart<T>> {
  using type = SimulateWishart<tag_t<T>>;
};

template<argument T>
struct peg_s<SimulateWishart<T>> {
  using type = SimulateWishart<peg_t<T>>;
};

template<argument T>
auto simulate_wishart(T&& x, const int n) {
  return SimulateWishart<tag_t<T>>{{tag(std::forward<T>(x)), n}};
}

}
