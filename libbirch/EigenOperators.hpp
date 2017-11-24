/**
 * @file
 *
 * Wrappers for Eigen operators that preserve its lazy evaluation.
 */
#pragma once

#include "libbirch/Eigen.hpp"
#include "libbirch/Array.hpp"

#include <type_traits>

#define UNARY_OPERATOR(op) \
  template<class Type, class Frame> \
  auto operator op(const bi::Array<Type,Frame>& x) { \
    return op x.toEigen(); \
  }

#define BINARY_OPERATOR(op) \
  template<class OtherType1, class Type2, class Frame2> \
  auto operator op(const OtherType1& x, const bi::Array<Type2,Frame2>& y) { \
    return x op y.toEigen(); \
  } \
  \
  template<class Type1, class Frame1, class OtherType2> \
  auto operator op(const bi::Array<Type1,Frame1>& x, const OtherType2& y) { \
    return x.toEigen() op y; \
  } \
  template<class Type1, class Frame1, class Type2, class Frame2> \
  auto operator op(const bi::Array<Type1,Frame1>& x, \
      const bi::Array<Type2,Frame2>& y) { \
    return x.toEigen() op y.toEigen(); \
  }

namespace bi {
UNARY_OPERATOR(+)
UNARY_OPERATOR(-)

BINARY_OPERATOR(+)
BINARY_OPERATOR(-)
BINARY_OPERATOR(*)
BINARY_OPERATOR(/)
}
