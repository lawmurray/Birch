/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<class Left, class Middle, class Right>
struct MatrixGather {
  BIRCH_TERNARY_FORM(MatrixGather, numbirch::gather)
  BIRCH_TERNARY_GRAD(numbirch::gather_grad)
  BIRCH_FORM
};

template<class Left, class Middle, class Right>
auto gather(const Left& l, const Middle& m, const Right& r) {
  return BIRCH_TERNARY_CONSTRUCT(MatrixGather);
}

}
