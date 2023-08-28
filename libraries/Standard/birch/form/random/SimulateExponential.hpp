/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct SimulateExponential : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::simulate_exponential(birch::eval(this->x));
  }

  int rows() const {
    return birch::rows(this->x);
  }

  int columns() const {
    return birch::columns(this->x);
  }
};

template<argument T>
struct is_form<SimulateExponential<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<SimulateExponential<T>> {
  using type = SimulateExponential<tag_t<T>>;
};

template<argument T>
struct peg_s<SimulateExponential<T>> {
  using type = SimulateExponential<peg_t<T>>;
};

template<argument T>
auto simulate_exponential(T&& x) {
  return SimulateExponential<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}
