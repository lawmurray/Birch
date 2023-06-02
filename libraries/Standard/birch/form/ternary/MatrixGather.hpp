/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<class Left, class Middle, class Right>
struct MatrixGather {
  BIRCH_TERNARY_FORM(MatrixGather)
};

BIRCH_TERNARY_SIZE(MatrixGather)
BIRCH_TERNARY(MatrixGather, numbirch::gather)
BIRCH_TERNARY_GRAD(MatrixGather, numbirch::gather_grad)

template<class Left, class Middle, class Right>
auto gather(const Left& l, const Middle& m, const Right& r) {
  return BIRCH_TERNARY_CONSTRUCT(MatrixGather);
}

}
