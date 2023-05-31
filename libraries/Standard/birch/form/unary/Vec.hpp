/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Vec {
  BIRCH_UNARY_FORM(Vec, numbirch::vec)
  BIRCH_UNARY_GRAD(numbirch::vec_grad)
  BIRCH_FORM
};

template<class Middle>
auto vec(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(Vec);
}

}
