/**
 * @file
 *
 * Wrappers for Eigen operators that preserve its lazy evaluation.
 */
#pragma once

#include "libbirch/Eigen.hpp"
#include "libbirch/Array.hpp"

#define UNARY_OPERATOR(op) \
  template<class Type, class Frame> \
  auto operator op(const libbirch::Array<Type,Frame>& x) { \
    return op x.toEigen(); \
  }

#define BINARY_OPERATOR(op) \
  template<class Type1, class Type2, class Frame2> \
  auto operator op(const Eigen::MatrixBase<Type1>& x, const libbirch::Array<Type2,Frame2>& y) { \
    return x op y.toEigen(); \
  } \
  \
  template<class Type1, class Frame1, class Type2> \
  auto operator op(const libbirch::Array<Type1,Frame1>& x, const Eigen::MatrixBase<Type2>& y) { \
    return x.toEigen() op y; \
  } \
  template<class Type1, class Frame1, class Type2, class Frame2> \
  auto operator op(const libbirch::Array<Type1,Frame1>& x, \
      const libbirch::Array<Type2,Frame2>& y) { \
    return x.toEigen() op y.toEigen(); \
  } \
  template<class Type1, class Frame1> \
  auto operator op(const Type1& x, const libbirch::Array<Type1,Frame1>& y) { \
    return (x op y.toEigen().array()).matrix(); \
  } \
  \
  template<class Type1, class Frame1> \
  auto operator op(const libbirch::Array<Type1,Frame1>& x, const Type1& y) { \
    return (x.toEigen().array() op y).matrix(); \
  } \
  template<class Type1, class Type2, typename = std::enable_if_t<std::is_same<Type1,typename Type2::value_type>::value>> \
  auto operator op(const Type1& x, const Eigen::MatrixBase<Type2>& y) { \
    return (x op y.array()).matrix(); \
  } \
  \
  template<class Type1, class Type2, typename = std::enable_if_t<std::is_same<Type2,typename Type1::value_type>::value>> \
  auto operator op(const Eigen::MatrixBase<Type1>& x, const Type2& y) { \
    return (x.array() op y).matrix(); \
  }

namespace bi {
UNARY_OPERATOR(+)
UNARY_OPERATOR(-)

BINARY_OPERATOR(+)
BINARY_OPERATOR(-)
BINARY_OPERATOR(*)
BINARY_OPERATOR(/)
}
