/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct VectorFill {
  BIRCH_UNARY_FORM(VectorFill, n)
};

BIRCH_UNARY(VectorFill, fill, n)
BIRCH_UNARY_GRAD(VectorFill, fill_grad, n)

template<argument Middle>
int rows(const VectorFill<Middle>& o) {
  return o.n;
}

template<argument Middle>
int columns(const VectorFill<Middle>& o) {
  return o.n;
}

template<argument Middle>
int length(const VectorFill<Middle>& o) {
  return o.n;
}

template<argument Middle>
int size(const VectorFill<Middle>& o) {
  return o.n;
}

}
