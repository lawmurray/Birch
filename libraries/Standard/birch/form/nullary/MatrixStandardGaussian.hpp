/**
 * @file
 */
#pragma once

#include "birch/form/Nullary.hpp"

namespace birch {

struct MatrixStandardGaussian {
  BIRCH_NULLARY_EVAL(MatrixStandardGaussian, standard_gaussian, R, C)
  BIRCH_NULLARY_GRAD(Abs)
  BIRCH_NULLARY_FORM(MatrixStandardGaussian, R, C)

  int rows() const {
    return R;
  }

  int columns() const {
    return C;
  }

  int size() const {
    return R*C;
  }
};

BIRCH_NULLARY_TYPE(MatrixStandardGaussian)
BIRCH_NULLARY_CALL(MatrixStandardGaussian, standard_gaussian, R, C)

}
