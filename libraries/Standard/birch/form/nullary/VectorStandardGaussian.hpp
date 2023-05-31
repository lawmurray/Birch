/**
 * @file
 */
#pragma once

#include "birch/form/Nullary.hpp"

namespace birch {

struct VectorStandardGaussian {
  BIRCH_NULLARY_FORM(VectorStandardGaussian, numbirch::standard_gaussian, n)
  BIRCH_NO_GRAD

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

inline auto standard_gaussian(const int n) {
  return BIRCH_NULLARY_CONSTRUCT(VectorStandardGaussian, n);
}

}
