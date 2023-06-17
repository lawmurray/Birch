/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct MatrixFill {
  BIRCH_UNARY_FORM(MatrixFill, R, C)
};

BIRCH_UNARY(MatrixFill, fill, R, C)
BIRCH_UNARY_GRAD(MatrixFill, fill_grad, R, C)

template<argument Middle>
int rows(const MatrixFill<Middle>& o) {
  return o.R;
}

template<argument Middle>
int columns(const MatrixFill<Middle>& o) {
  return o.C;
}

template<argument Middle>
int length(const MatrixFill<Middle>& o) {
  return o.R;
}

template<argument Middle>
int size(const MatrixFill<Middle>& o) {
  return o.R*o.C;
}

}
