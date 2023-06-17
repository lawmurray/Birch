/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Iota {
  BIRCH_UNARY_FORM(Iota, n)
};

BIRCH_UNARY_SIZE(Iota)
BIRCH_UNARY(Iota, iota, n)
BIRCH_UNARY_GRAD(Iota, iota_grad, n)

}
