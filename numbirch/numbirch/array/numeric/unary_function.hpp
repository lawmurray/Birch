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
 * @tparam T Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> abs(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  abs(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Arc cosine.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> acos(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  acos(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Arc sine.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> asin(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  asin(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Arc tangent.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> atan(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  atan(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Round to smallest integer value not less than argument.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> ceil(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  ceil(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Inverse of a symmetric positive definite square matrix, via the Cholesky
 * factorization.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * 
 * @param S Symmetric positive definite matrix.
 * 
 * @return Matrix.
 */
template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
PURE Matrix<T> cholinv(const Matrix<T>& S) {
  Matrix<T> B(make_shape(S.rows(), S.columns()));
  cholinv(S.rows(), S.data(), S.stride(), B.data(), B.stride());
  return B;
}

/**
 * Cosine.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> cos(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  cos(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Hyperbolic cosine.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> cosh(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  cosh(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Count of non-zero elements.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return Count of non-zero elements in the array.
 */
template<class T, int D, std::enable_if_t<
    std::is_arithmetic<T>::value,int> = 0>
PURE Scalar<int> count(const Array<T,D>& A) {
  Scalar<int> b;
  count(A.width(), A.height(), A.data(), A.stride(), b.data());
  return b;
}

/**
 * Construct diagonal matrix. Diagonal elements are assigned to a given scalar
 * value, while all off-diagonal elements are assigned zero.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * 
 * @param x Scalar to assign to diagonal.
 * @param n Number of rows and columns.
 */
template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
PURE Matrix<T> diagonal(const Scalar<T>& x, const int n) {
  Matrix<T> B(make_shape(n, n));
  diagonal(x.data(), n, B.data(), B.stride());
  return B;
}

/**
 * Construct diagonal matrix. Diagonal elements are assigned to a given scalar
 * value, while all off-diagonal elements are assigned zero.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * 
 * @param x Scalar to assign to diagonal.
 * @param n Number of rows and columns.
 */
template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
PURE Matrix<T> diagonal(const T& x, const int n) {
  return diagonal(Scalar<T>(x), n);
}

/**
 * Digamma.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> digamma(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  digamma(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Exponential.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> exp(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  exp(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Exponential minus one.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> expm1(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  expm1(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Round to largest integer value not greater than argument.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> floor(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  floor(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Inverse of a square matrix.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix.
 * 
 * @return Matrix.
 */
template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
PURE Matrix<T> inv(const Matrix<T>& A) {
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
 * @tparam T Floating point type.
 * 
 * @param S Symmetric positive definite matrix.
 * 
 * @return Logarithm of the determinant of `S`.
 */
template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
PURE Scalar<T> lcholdet(const Matrix<T>& S) {
  Scalar<T> b;
  lcholdet(S.rows(), S.data(), S.stride(), b.data());
  return b;
}

/**
 * Logarithm of the absolute value of the determinant of a square matrix.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix.
 * 
 * @return Logarithm of the absolute value of the determinant of `A`.
 */
template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
PURE Scalar<T> ldet(const Matrix<T>& A) {
  Scalar<T> b;
  ldet(A.rows(), A.data(), A.stride(), b.data());
  return b;
}

/**
 * Logarithm of the factorial function.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 * 
 * @note The return type `T` must be explicitly specified.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> lfact(const Array<int,D>& A) {
  Array<T,D> B(A.shape().compact());
  lfact(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Gradient of lfact().
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 * 
 * @note The return type `T` must be explicitly specified.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> lfact_grad(const Array<T,D>& G, const Array<int,D>& A) {
  Array<T,D> B(A.shape().compact());
  lfact_grad(A.width(), A.height(), G.data(), G.stride(), A.data(),
      A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Logarithm of the gamma function.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> lgamma(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  lgamma(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Logarithm.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> log(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  log(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Logarithm of one plus argument.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> log1p(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  log1p(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Reciprocal. For element @f$(i,j)@f$, computes @f$B_{ij} = 1/A_{ij}@f$. The
 * division is as for the type `T`; this will always return zero for an
 * integer type.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> rcp(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  rcp(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Rectification. For element @f$(i,j)@f$, computes @f$B_{ij} = \max(A_{ij},
 * 0)@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> rectify(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  rectify(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Gradient of rectify().
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> rectify_grad(const Array<T,D>& G, const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  rectify_grad(A.width(), A.height(), G.data(), G.stride(), A.data(),
      A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Round to nearest integer value.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> round(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  round(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Sine.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> sin(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  sin(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Construct single-entry vector. One of the elements of the vector is one,
 * all others are zero.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param i Index of single entry (1-based).
 * @param n Length of vector.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
PURE Vector<T> single(const Scalar<int>& i, const int n) {
  Vector<T> x(make_shape(n));
  single(i.data(), n, x.data(), x.stride());
  return x;
}

/**
 * Construct single-entry vector. One of the elements of the vector is one,
 * all others are zero.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * 
 * @param i Index of single entry (1-based).
 * @param n Length of vector.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
PURE Vector<T> single(const int& i, const int n) {
  return single<T>(Scalar<int>(i), n);
}

/**
 * Hyperbolic sine.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> sinh(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  sinh(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Square root.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> sqrt(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  sqrt(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Sum of elements.
 * 
 * @ingroup array
 * 
 * @tparam T Arithmetic type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return Sum of elements of the array.
 */
template<class T, int D, std::enable_if_t<
    std::is_arithmetic<T>::value,int> = 0>
PURE Scalar<T> sum(const Array<T,D>& A) {
  Scalar<T> b;
  sum(A.width(), A.height(), A.data(), A.stride(), b.data());
  return b;
}

/**
 * Tangent.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> tan(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  tan(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Hyperbolic tangent.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * 
 * @return %Array.
 */
template<class T, int D, std::enable_if_t<
    std::is_floating_point<T>::value,int> = 0>
PURE Array<T,D> tanh(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  tanh(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Matrix trace.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix.
 * 
 * @return Trace of the matrix.
 */
template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
PURE Scalar<T> trace(const Matrix<T>& A) {
  Scalar<T> b;
  trace(A.rows(), A.columns(), A.data(), A.stride(), b.data());
  return b;
}

/**
 * Scalar product and transpose. Computes @f$B = xA^\top@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Floating point type.
 * 
 * @param A Matrix.
 * 
 * @return Matrix.
 */
template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
PURE Matrix<T> transpose(const Matrix<T>& A) {
  Matrix<T> B(make_shape(A.columns(), A.rows()));
  transpose(B.rows(), B.columns(), A.data(), A.stride(), B.data(),
      B.stride());
  return B;
}

}
