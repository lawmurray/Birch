/**
 * @file
 */
#pragma once

namespace bi {
class FuncParameter;
class FuncReference;

/**
 * Comparison of two functions by the specialisation of their signatures.
 */
struct signature_less_equal {
  bool operator()(FuncParameter* o1, FuncParameter* o2);
  bool operator()(const FuncReference* o1, FuncParameter* o2);
};
}
