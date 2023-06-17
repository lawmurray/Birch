/**
 * @file
 */
#pragma once

#include "birch/form/Nullary.hpp"

namespace birch {

struct MatrixStandardGaussian {
  BIRCH_NULLARY_FORM(MatrixStandardGaussian, R, C)
};

inline int rows(const MatrixStandardGaussian& o) {
  return o.R;
}

inline int columns(const MatrixStandardGaussian& o) {
  return o.C;
}

inline int length(const MatrixStandardGaussian& o) {
  return o.R;
}

inline int size(const MatrixStandardGaussian& o) {
  return o.R*o.C;
}

BIRCH_NULLARY(MatrixStandardGaussian, standard_gaussian, R, C)

}
