/**
 * @file
 */
#pragma once

#include "birch/form/Binary.hpp"

namespace birch {

template<argument Left, argument Right>
struct Equal {
  BIRCH_BINARY_FORM(Equal)
};

BIRCH_BINARY_SIZE(Equal)
BIRCH_BINARY(Equal, equal)
BIRCH_BINARY_GRAD(Equal, equal_grad)

template<argument Left, argument Right>
requires (!numbirch::arithmetic<Left> || !numbirch::arithmetic<Right>)
auto operator==(Left&& l, Right&& r) {
  return equal(std::forward<Left>(l), std::forward<Right>(r));
}

}
