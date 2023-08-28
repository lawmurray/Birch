/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T, argument U>
struct SimulateGamma : public Form<T,U> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::simulate_gamma(birch::eval(this->x), birch::eval(this->y));
  }

  int rows() const {
    return birch::rows(this->x, this->y);
  }

  int columns() const {
    return birch::columns(this->x, this->y);
  }
};

template<argument T, argument U>
struct is_form<SimulateGamma<T,U>> {
  static constexpr bool value = true;
};

template<argument T, argument U>
struct tag_s<SimulateGamma<T,U>> {
  using type = SimulateGamma<tag_t<T>,tag_t<U>>;
};

template<argument T, argument U>
struct peg_s<SimulateGamma<T,U>> {
  using type = SimulateGamma<peg_t<T>,peg_t<U>>;
};

template<argument T, argument U>
auto simulate_gamma(T&& x, U&& y) {
  return SimulateGamma<tag_t<T>,tag_t<U>>{{tag(std::forward<T>(x)),
      tag(std::forward<U>(y))}};
}

}
