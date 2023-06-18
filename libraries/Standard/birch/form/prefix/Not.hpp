/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Not {
  BIRCH_UNARY_FORM(Not)
  BIRCH_UNARY_SIZE(Not)
  BIRCH_UNARY_EVAL(Not, logical_not)
  BIRCH_UNARY_GRAD(Not, logical_not_grad)
};

BIRCH_UNARY_TYPE(Not)
BIRCH_UNARY_CALL(Not, logical_not)

template<argument Middle>
requires (!numbirch::arithmetic<Middle>)
auto operator!(Middle&& m) {
  return logical_not(std::forward<Middle>(m));
}

}
