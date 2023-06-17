/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Mat {
  BIRCH_UNARY_FORM(Mat, n)
};

BIRCH_UNARY_SIZE(Mat)
BIRCH_UNARY(Mat, mat, n)
BIRCH_UNARY_GRAD(Mat, mat_grad, n)

}
