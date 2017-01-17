/**
 * @file
 */
#pragma once

#include "bi/primitive/possibly.hpp"

namespace bi {
class FuncParameter;
class FuncReference;

/**
 * Comparison of two functions by the specialisation of their signatures.
 */
struct signature_less_equal {
  possibly operator()(FuncParameter* o1, FuncParameter* o2);
  possibly operator()(FuncReference* o1, FuncParameter* o2);
};
}
