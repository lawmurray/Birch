/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct SimulateBernoulli : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::simulate_bernoulli(birch::eval(this->x));
  }

  int rows() const {
    return birch::rows(this->x);
  }

  int columns() const {
    return birch::columns(this->x);
  }
};

template<argument T>
struct is_form<SimulateBernoulli<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<SimulateBernoulli<T>> {
  using type = SimulateBernoulli<tag_t<T>>;
};

template<argument T>
struct peg_s<SimulateBernoulli<T>> {
  using type = SimulateBernoulli<peg_t<T>>;
};

template<argument T>
auto simulate_bernoulli(T&& x) {
  return SimulateBernoulli<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}
