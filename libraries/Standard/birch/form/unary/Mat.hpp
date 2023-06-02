/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Mat {
  BIRCH_UNARY_FORM(Mat, n)
};

BIRCH_UNARY_SIZE(Mat)
BIRCH_UNARY(Mat, numbirch::mat, n)
BIRCH_UNARY_GRAD(Mat, numbirch::mat_grad, n)

template<class Middle>
auto mat(const Middle& m, const int n) {
  return BIRCH_UNARY_CONSTRUCT(Mat, n);
}

}
