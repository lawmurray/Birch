/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct VectorFill {
  BIRCH_UNARY_FORM(VectorFill, n)
};

BIRCH_UNARY(VectorFill, numbirch::fill, n)
BIRCH_UNARY_GRAD(VectorFill, numbirch::fill_grad, n)

template<class Middle>
int rows(const VectorFill<Middle>& o) {
  return o.n;
}

template<class Middle>
int columns(const VectorFill<Middle>& o) {
  return o.n;
}

template<class Middle>
int length(const VectorFill<Middle>& o) {
  return o.n;
}

template<class Middle>
int size(const VectorFill<Middle>& o) {
  return o.n;
}

template<class Middle>
auto fill(const Middle& m, const int n) {
  return BIRCH_UNARY_CONSTRUCT(VectorFill, n);
}

}
