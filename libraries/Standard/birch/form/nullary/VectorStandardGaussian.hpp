/**
 * @file
 */
#pragma once

#include "birch/form/Nullary.hpp"

namespace birch {

struct VectorStandardGaussian {
  BIRCH_NULLARY_FORM(VectorStandardGaussian, n)
  BIRCH_NULLARY_EVAL(VectorStandardGaussian, standard_gaussian, n)

  int rows() const {
    return n;
  }

  int columns() const {
    return n;
  }

  int length() const {
    return n;
  }

  int size() const {
    return n;
  }
};

BIRCH_NULLARY_TYPE(VectorStandardGaussian)
BIRCH_NULLARY_CALL(VectorStandardGaussian, standard_gaussian, m)

}
