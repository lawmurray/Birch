/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct VectorFill {
  BIRCH_UNARY_FORM(VectorFill, numbirch::fill, n)
  BIRCH_UNARY_GRAD(numbirch::fill_grad, n)

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

template<class Middle>
auto fill(const Middle& m, const int n) {
  return BIRCH_UNARY_CONSTRUCT(VectorFill, n);
}

}
