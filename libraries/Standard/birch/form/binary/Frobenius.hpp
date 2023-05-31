/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<class Left, class Right>
struct Frobenius {
  BIRCH_BINARY_FORM(Frobenius, numbirch::frobenius)
  BIRCH_BINARY_GRAD(numbirch::frobenius_grad)
  BIRCH_FORM
};

template<class Left, class Right>
auto frobenius(const Left& l, const Right& r) {
  return BIRCH_BINARY_CONSTRUCT(Frobenius);
}

}
