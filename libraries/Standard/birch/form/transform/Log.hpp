/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<argument T>
struct Log : public Form<T> {
  BIRCH_FORM

  auto eval() const {
    return numbirch::log(birch::eval(this->x));
  }

  template<numbirch::numeric G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    if (!birch::is_constant(this->x)) {
      birch::shallow_grad(this->x, numbirch::log_grad(std::forward<G>(g),
          birch::eval(this->x)), visitor);
    }
  }

  int rows() const {
    return birch::rows(this->x);
  }

  int columns() const {
    return birch::columns(this->x);
  }
};

template<argument T>
struct is_form<Log<T>> {
  static constexpr bool value = true;
};

template<argument T>
struct tag_s<Log<T>> {
  using type = Log<tag_t<T>>;
};

template<argument T>
struct peg_s<Log<T>> {
  using type = Log<peg_t<T>>;
};

template<argument T>
auto log(T&& x) {
  return Log<tag_t<T>>{{tag(std::forward<T>(x))}};
}

}

#include "birch/form/transform/Exp.hpp"

namespace birch {

template<argument T>
decltype(auto) log(const Exp<T>& x) {
  return x.x;
}

}
