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
  \
  template<class T, class U, class G> \
  auto operator op(const Eigen::DiagonalWrapper<T>& x, const libbirch::Array<U,G>& y) { \
    return x op y.toEigen(); \
  } \
  \
  template<class T, class F, class U> \
  auto operator op(const libbirch::Array<T,F>& x, const Eigen::DiagonalWrapper<U>& y) { \
    return x.toEigen() op y; \
  } \
  \
  template<class T, unsigned Mode, class U, class G> \
  auto operator op(const Eigen::TriangularView<T,Mode>& x, const libbirch::Array<U,G>& y) { \
    return x op y.toEigen(); \
  } \
  \
  template<class T, class F, class U, unsigned Mode> \
  auto operator op(const libbirch::Array<T,F>& x, const Eigen::TriangularView<U,Mode>& y) { \
    return x.toEigen() op y; \
  } \
  \
  template<class T, class F, class U, class G> \
  auto operator op(const libbirch::Array<T,F>& x, const libbirch::Array<U,G>& y) { \
    return x.toEigen() op y.toEigen(); \
  }

/**
 * A binary operator with a scalar on the left.
 */
#define LEFT_SCALAR_BINARY_OPERATOR(op) \
  template<class T, class F> \
  auto operator op(const T& x, const libbirch::Array<T,F>& y) { \
    return x op y.toEigen(); \
  }

/**
 * A binary operator with a scalar on the right.
 */
#define RIGHT_SCALAR_BINARY_OPERATOR(op) \
  template<class T, class F> \
  auto operator op(const libbirch::Array<T,F>& x, const T& y) { \
    return x.toEigen() op y; \
  }

namespace birch {
UNARY_OPERATOR(+)
UNARY_OPERATOR(-)
BINARY_OPERATOR(+)
BINARY_OPERATOR(-)
BINARY_OPERATOR(*)
BINARY_OPERATOR(==)
BINARY_OPERATOR(!=)
LEFT_SCALAR_BINARY_OPERATOR(*)
RIGHT_SCALAR_BINARY_OPERATOR(*)
RIGHT_SCALAR_BINARY_OPERATOR(/)

/*
 * Some specific-case extras.
 */
template<class T, class U> \
auto operator==(const Eigen::LLT<T>& x, const Eigen::LLT<U>& y) {
  return x.reconstructedMatrix() == y.reconstructedMatrix();
}

template<class T, class U> \
auto operator!=(const Eigen::LLT<T>& x, const Eigen::LLT<U>& y) { \
  return x.reconstructedMatrix() != y.reconstructedMatrix();
}

}
