/**
 * @file
 */
#pragma once

#include "numbirch/array/Array.hpp"
#include "numbirch/array/Scalar.hpp"
#include "numbirch/array/Vector.hpp"
#include "numbirch/array/Matrix.hpp"
#include "numbirch/memory.hpp"
#include "numbirch/numeric.hpp"
#include "numbirch/function.hpp"

#include <cmath>

namespace numbirch {
/**
 * Absolute value.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double`, `float` or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<T,D> abs(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  abs(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Arc cosine.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto acos(const Array<T,D>& A) {
  Array<decltype(std::acos(T())),D> B(A.shape().compact());
  acos(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Arc sine.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto asin(const Array<T,D>& A) {
  Array<decltype(std::asin(T())),D> B(A.shape().compact());
  asin(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Arc tangent.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto atan(const Array<T,D>& A) {
  Array<decltype(std::atan(T())),D> B(A.shape().compact());
  atan(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Round to smallest integer value not less than argument.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto ceil(const Array<T,D>& A) {
  Array<decltype(ceil(T())),D> B(A.shape().compact());
  ceil(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Inverse of a symmetric positive definite square matrix, via the Cholesky
 * factorization.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param S Symmetric positive definite matrix.
 * 
 * @return Matrix.
 */
template<class T>
Matrix<T> cholinv(const Matrix<T>& S) {
  Matrix<T> B(make_shape(S.rows(), S.columns()));
  cholinv(S.rows(), S.data(), S.stride(), B.data(), B.stride());
  return B;
}

/**
 * Cosine.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto cos(const Array<T,D>& A) {
  Array<decltype(std::cos(T())),D> B(A.shape().compact());
  cos(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Hyperbolic cosine.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto cosh(const Array<T,D>& A) {
  Array<decltype(std::cos(T())),D> B(A.shape().compact());
  cosh(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Digamma.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto digamma(const Array<T,D>& A) {
  Array<decltype(digamma(T())),D> B(A.shape().compact());
  digamma(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Exponential.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto exp(const Array<T,D>& A) {
  Array<decltype(std::exp(T())),D> B(A.shape().compact());
  exp(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Exponential minus one.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto expm1(const Array<T,D>& A) {
  Array<decltype(std::expm1(T())),D> B(A.shape().compact());
  expm1(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Round to largest integer value not greater than argument.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto floor(const Array<T,D>& A) {
  Array<decltype(floor(T())),D> B(A.shape().compact());
  floor(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Inverse of a square matrix.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Matrix<T> inv(const Matrix<T>& A) {
  Matrix<T> B(make_shape(A.rows(), A.columns()));
  inv(A.rows(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Logarithm of the determinant of a symmetric positive definite matrix, via
 * the Cholesky factorization. The determinant of a positive definite matrix
 * is always positive.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param S Symmetric positive definite matrix.
 * 
 * @return Logarithm of the determinant of `S`.
 */
template<class T>
Scalar<T> lcholdet(const Matrix<T>& S) {
  Scalar<T> b;
  lcholdet(S.rows(), S.data(), S.stride(), b.data());
  return b;
}

/**
 * Logarithm of the absolute value of the determinant of a square matrix.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * 
 * @return Logarithm of the absolute value of the determinant of `A`.
 */
template<class T>
Scalar<T> ldet(const Matrix<T>& A) {
  Scalar<T> b;
  ldet(A.rows(), A.data(), A.stride(), b.data());
  return b;
}

/**
 * Logarithm of the gamma function.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto lgamma(const Array<T,D>& A) {
  Array<decltype(std::lgamma(T())),D> B(A.shape().compact());
  lgamma(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Logarithm.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto log(const Array<T,D>& A) {
  Array<decltype(std::log(T())),D> B(A.shape().compact());
  log(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Logarithm of one plus argument.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto log1p(const Array<T,D>& A) {
  Array<decltype(std::log1p(T())),D> B(A.shape().compact());
  log1p(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Rectification. For element @f$(i,j)@f$, computes @f$B_{ij} = \max(A_{ij},
 * 0)@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto rectify(const Array<T,D>& A) {
  Array<decltype(rectify(T())),D> B(A.shape().compact());
  rectify(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Round to nearest integer value.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto round(const Array<T,D>& A) {
  Array<decltype(round(T())),D> B(A.shape().compact());
  round(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Sine.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto sin(const Array<T,D>& A) {
  Array<decltype(std::sin(T())),D> B(A.shape().compact());
  sin(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Hyperbolic sine.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto sinh(const Array<T,D>& A) {
  Array<decltype(std::sinh(T())),D> B(A.shape().compact());
  sinh(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Square root.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto sqrt(const Array<T,D>& A) {
  Array<decltype(std::sqrt(T())),D> B(A.shape().compact());
  sqrt(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Sum of elements.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double`, `float` or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return Sum of elements of the array.
 */
template<class T, int D>
Scalar<T> sum(const Array<T,D>& A) {
  Scalar<T> b;
  sum(A.width(), A.height(), A.data(), A.stride(), b.data());
  return b;
}

/**
 * Tangent.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto tan(const Array<T,D>& A) {
  Array<decltype(std::tanh(T())),D> B(A.shape().compact());
  tan(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Hyperbolic tangent.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
auto tanh(const Array<T,D>& A) {
  Array<decltype(std::tanh(T())),D> B(A.shape().compact());
  tanh(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Matrix trace.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * 
 * @return Trace of the matrix.
 */
template<class T>
Scalar<T> trace(const Matrix<T>& A) {
  Scalar<T> b;
  trace(A.rows(), A.columns(), A.data(), A.stride(), b.data());
  return b;
}

/**
 * Scalar product and transpose. Computes @f$B = xA^\top@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param x Scalar.
 * @param A Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Matrix<T> transpose(const Matrix<T>& A) {
  Matrix<T> B(make_shape(A.columns(), A.rows()));
  transpose(B.rows(), B.columns(), A.data(), A.stride(), B.data(),
      B.stride());
  return B;
}

}
