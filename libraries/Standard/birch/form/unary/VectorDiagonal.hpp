/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct VectorDiagonal {
  BIRCH_UNARY_FORM(VectorDiagonal)
};

BIRCH_UNARY(VectorDiagonal, diagonal)
BIRCH_UNARY_GRAD(VectorDiagonal, diagonal_grad)

template<argument Middle>
int rows(const VectorDiagonal<Middle>& o) {
  return length(o.m);
}

template<argument Middle>
int columns(const VectorDiagonal<Middle>& o) {
  return length(o.m);
}

template<argument Middle>
int length(const VectorDiagonal<Middle>& o) {
  return length(o.m);
}

template<argument Middle>
int size(const VectorDiagonal<Middle>& o) {
  return pow(length(o.m), 2);
}

}
