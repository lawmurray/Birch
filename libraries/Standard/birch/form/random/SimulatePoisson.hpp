/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct SimulatePoisson : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::simulate_poisson(birch::eval(this->x));
  }

  int rows() const {
    return birch::rows(this->x);
  }

  int columns() const {
    return birch::columns(this->x);
  }
};

template<argument T>
struct is_form<SimulatePoisson<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<SimulatePoisson<T>> {
  using type = SimulatePoisson<tag_t<T>>;
};

template<argument T>
struct peg_s<SimulatePoisson<T>> {
  using type = SimulatePoisson<peg_t<T>>;
};

template<argument T>
auto simulate_poisson(T&& x) {
  return SimulatePoisson<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}
