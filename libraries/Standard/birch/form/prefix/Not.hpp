/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Not {
  BIRCH_UNARY_FORM(Not)
};

BIRCH_UNARY_SIZE(Not)
BIRCH_UNARY(Not, logical_not)
BIRCH_UNARY_GRAD(Not, logical_not_grad)

template<argument Middle>
requires (!numbirch::arithmetic<Middle>)
auto operator!(Middle&& m) {
  return logical_not(std::forward<Middle>(m));
}

}
