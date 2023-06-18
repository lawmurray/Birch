/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Equal {
  BIRCH_BINARY_FORM(Equal)
  BIRCH_BINARY_SIZE(Equal)
  BIRCH_BINARY_EVAL(Equal, equal)
  BIRCH_BINARY_GRAD(Equal, equal_grad)
};

BIRCH_BINARY_TYPE(Equal)
BIRCH_BINARY_CALL(Equal, equal)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator==(Left&& l, Right&& r) {
  return equal(std::forward<Left>(l), std::forward<Right>(r));
}

}
