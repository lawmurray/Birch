/**
 * @file
 */
#pragma once

#include "birch/form/Nullary.hpp"

namespace birch {

struct MatrixStandardGaussian {
  BIRCH_NULLARY_FORM(MatrixStandardGaussian, numbirch::standard_gaussian, R, C)
  BIRCH_NO_GRAD

  int rows() const {
    return R;
  }

  int columns() const {
    return C;
  }

  int length() const {
    return R;
  }

  int size() const {
    return R*C;
  }
};

inline auto standard_gaussian(const int R, const int C) {
  return BIRCH_NULLARY_CONSTRUCT(MatrixStandardGaussian, R, C);
}

}
