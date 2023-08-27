/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct StandardGaussianOp {
  static auto eval(const int n) {
    return numbirch::standard_gaussian(n);
  }

  static int rows(const int n) {
    return n;
  }

  static constexpr int columns(const int n) {
    return 1;
  }

  static auto eval(const int m, const int n) {
    return numbirch::standard_gaussian(m, n);
  }

  static int rows(const int m, const int n) {
    return m;
  }

  static int columns(const int m, const int n) {
    return n;
  }
};

template<argument... Args>
using StandardGaussian = Form<StandardGaussianOp,Args...>;

inline auto standard_gaussian(const int n) {
  return StandardGaussian<int>(std::in_place, n);
}

inline auto standard_gaussian(const int m, const int n) {
  return StandardGaussian<int,int>(std::in_place, m, n);
}

}
