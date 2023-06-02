/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct VectorDiagonal {
  BIRCH_UNARY_FORM(VectorDiagonal)
};

BIRCH_UNARY(VectorDiagonal, numbirch::diagonal)
BIRCH_UNARY_GRAD(VectorDiagonal, numbirch::diagonal_grad)

template<class Middle>
int rows(const VectorDiagonal<Middle>& o) {
  return length(o.m);
}

template<class Middle>
int columns(const VectorDiagonal<Middle>& o) {
  return length(o.m);
}

template<class Middle>
int length(const VectorDiagonal<Middle>& o) {
  return length(o.m);
}

template<class Middle>
int size(const VectorDiagonal<Middle>& o) {
  return pow(length(o.m), 2);
}

template<class Middle>
auto diagonal(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(VectorDiagonal);
}

}
