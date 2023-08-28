/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct SimulateDirichlet : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::simulate_dirichlet(birch::eval(this->x));
  }

  int rows() const {
    return birch::rows(this->x);
  }

  int columns() const {
    return birch::columns(this->x);
  }
};

template<argument T>
struct is_form<SimulateDirichlet<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<SimulateDirichlet<T>> {
  using type = SimulateDirichlet<tag_t<T>>;
};

template<argument T>
struct peg_s<SimulateDirichlet<T>> {
  using type = SimulateDirichlet<peg_t<T>>;
};

template<argument T>
auto simulate_dirichlet(T&& x) {
  return SimulateDirichlet<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}
