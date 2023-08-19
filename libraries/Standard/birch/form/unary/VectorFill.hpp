/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct VectorFill {
  BIRCH_UNARY_FORM(VectorFill, n)
  BIRCH_UNARY_EVAL(VectorFill, fill, n)
  BIRCH_UNARY_GRAD(VectorFill, fill_grad, n)

  int rows() const {
    return n;
  }

  int columns() const {
    return n;
  }

  int size() const {
    return n;
  }
};

BIRCH_UNARY_TYPE(VectorFill)
BIRCH_UNARY_CALL(VectorFill, fill, n)

}
