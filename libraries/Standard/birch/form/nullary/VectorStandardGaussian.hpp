/**
 * @file
 */
#pragma once

#include "birch/form/Nullary.hpp"

namespace birch {

struct VectorStandardGaussian {
  BIRCH_NULLARY_FORM(VectorStandardGaussian, n)
};

inline int rows(const VectorStandardGaussian& o) {
  return o.n;
}

inline int columns(const VectorStandardGaussian& o) {
  return o.n;
}

inline int length(const VectorStandardGaussian& o) {
  return o.n;
}

inline int size(const VectorStandardGaussian& o) {
  return o.n;
}

BIRCH_NULLARY(VectorStandardGaussian, standard_gaussian, n)

}
