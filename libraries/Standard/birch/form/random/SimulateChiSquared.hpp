/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct SimulateChiSquared : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::simulate_chi_squared(birch::eval(this->x));
  }

  int rows() const {
    return birch::rows(this->x);
  }

  int columns() const {
    return birch::columns(this->x);
  }
};

template<argument T>
struct is_form<SimulateChiSquared<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<SimulateChiSquared<T>> {
  using type = SimulateChiSquared<tag_t<T>>;
};

template<argument T>
struct peg_s<SimulateChiSquared<T>> {
  using type = SimulateChiSquared<peg_t<T>>;
};

template<argument T>
auto simulate_chi_squared(T&& x) {
  return SimulateChiSquared<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}
