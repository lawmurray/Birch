/**
 * @file
 *
 * Wrappers for Eigen operators that preserve its lazy evaluation.
 */
#pragma once

#include "libbirch/Eigen.hpp"
#include "libbirch/Array.hpp"

/*
 * A unary operator.
 */
#define UNARY_OPERATOR(op) \
  template<class T, class F> \
  auto operator op(const libbirch::Array<T,F>& x) { \
    return op x.toEigen(); \
  }

/*
 * A binary operator.
 */
#define BINARY_OPERATOR(op) \
  template<class T, class U, class G> \
  auto operator op(const Eigen::MatrixBase<T>& x, const libbirch::Array<U,G>& y) { \
    return x op y.toEigen(); \
  } \
  \
  template<class T, class F, class U> \
  auto operator op(const libbirch::Array<T,F>& x, const Eigen::MatrixBase<U>& y) { \
    return x.toEigen() op y; \
  } \
  template<class T, class F, class U, class G> \
  auto operator op(const libbirch::Array<T,F>& x, \
      const libbirch::Array<U,G>& y) { \
    return x.toEigen() op y.toEigen(); \
  }

/**
 * A binary operator with a scalar on the left.
 */
#define LEFT_SCALAR_BINARY_OPERATOR(op) \
  template<class T, class F> \
  auto operator op(const T& x, const libbirch::Array<T,F>& y) { \
    return (x op y.toEigen().array()).matrix(); \
  }

/**
 * A binary operator with a scalar on the right.
 */
#define RIGHT_SCALAR_BINARY_OPERATOR(op) \
  template<class T, class F> \
  auto operator op(const libbirch::Array<T,F>& x, const T& y) { \
    return (x.toEigen().array() op y).matrix(); \
  }

/**
 * A binary operator with a scalar on the left that Eigen does not define
 * itself.
 */
#define LEFT_EXTRA_SCALAR_BINARY_OPERATOR(op) \
  template<class T, class U, typename = std::enable_if_t<std::is_same<T,typename U::value_type>::value>> \
  auto operator op(const T& x, const Eigen::MatrixBase<U>& y) { \
    return (x op y.array()).matrix(); \
  }

/**
 * A binary operator with a scalar on the right that Eigen does not define
 * itself.
 */
#define RIGHT_EXTRA_SCALAR_BINARY_OPERATOR(op) \
  template<class T, class U, typename = std::enable_if_t<std::is_same<U,typename T::value_type>::value>> \
  auto operator op(const Eigen::MatrixBase<T>& x, const U& y) { \
    return (x.array() op y).matrix(); \
  }

namespace bi {
UNARY_OPERATOR(+)
UNARY_OPERATOR(-)

BINARY_OPERATOR(+)
BINARY_OPERATOR(-)
BINARY_OPERATOR(*)

LEFT_SCALAR_BINARY_OPERATOR(+)
LEFT_SCALAR_BINARY_OPERATOR(-)
LEFT_SCALAR_BINARY_OPERATOR(*)

RIGHT_SCALAR_BINARY_OPERATOR(+)
RIGHT_SCALAR_BINARY_OPERATOR(-)
RIGHT_SCALAR_BINARY_OPERATOR(*)
RIGHT_SCALAR_BINARY_OPERATOR(/)

LEFT_EXTRA_SCALAR_BINARY_OPERATOR(+)
LEFT_EXTRA_SCALAR_BINARY_OPERATOR(-)

RIGHT_EXTRA_SCALAR_BINARY_OPERATOR(+)
RIGHT_EXTRA_SCALAR_BINARY_OPERATOR(-)
}
