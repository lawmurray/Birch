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

namespace numbirch {
/**
 * Copy sign of a number.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<T,D> copysign(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<T,D> C(A.shape().compact());
  copysign(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Hadamard (element-wise) matrix multiplication.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<T,D> hadamard(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<T,D> C(A.shape().compact());
  hadamard(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Lower-triangular Cholesky factor of a matrix multiplied by a vector.
 * Computes @f$y = Lx@f$, where @f$S = LL^\top@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param S Symmetric positive definite matrix.
 * @param x Vector.
 * 
 * @return Vector.
 */
template<class T>
Array<T,1> cholmul(const Array<T,2>& S, const Array<T,1>& x) {
  assert(S.rows() == S.columns());
  assert(S.columns() == x.length());

  Array<T,1> y(make_shape(S.rows()));
  cholmul(y.rows(), S.data(), S.stride(), x.data(), x.stride(), y.data(),
      y.stride());
  return y;
}

/**
 * Lower-triangular Cholesky factor of a matrix multiplied by a matrix.
 * Computes @f$C = LB@f$, where @f$S = LL^\top@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param S Symmetric positive definite matrix.
 * @param B Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Array<T,2> cholmul(const Array<T,2>& S, const Array<T,2>& B) {
  assert(S.rows() == S.columns());
  assert(S.columns() == S.rows());

  Array<T,2> C(make_shape(S.rows(), B.columns()));
  cholmul(C.rows(), C.columns(), S.data(), S.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Outer product of matrix and lower-triangular Cholesky factor of another
 * matrix. Computes @f$C = AL^\top@f$, where @f$S = LL^\top@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * @param S Symmetric positive definite matrix.
 * 
 * @return Matrix.
 */
template<class T>
Array<T,2> cholouter(const Array<T,2>& A, const Array<T,2>& S) {
  assert(A.columns() == S.columns());
  assert(S.rows() == S.columns());

  Array<T,2> C(make_shape(A.rows(), S.rows()));
  cholouter(C.rows(), C.columns(), A.data(), A.stride(), S.data(), S.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Matrix-vector solve, via the Cholesky factorization. Solves for @f$x@f$ in
 * @f$Sx = y@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param S Symmetric positive definite matrix.
 * @param y Vector.
 * 
 * @return Vector.
 */
template<class T>
Array<T,1> cholsolve(const Array<T,2>& S, const Array<T,1>& y) {
  assert(S.rows() == S.columns());
  assert(S.rows() == y.length());

  Array<T,1> x(make_shape(y.length()));
  cholsolve(x.rows(), S.data(), S.stride(), x.data(), x.stride(), y.data(),
      y.stride());
  return x;
}

/**
 * Matrix-matrix solve, via the Cholesky factorization. Solves for @f$A@f$ in
 * @f$SA = B@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param S Symmetric positive definite matrix.
 * @param B Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Array<T,2> cholsolve(const Array<T,2>& S, const Array<T,2>& B) {
  assert(S.rows() == S.columns());
  assert(S.rows() == B.rows());

  Array<T,2> A(make_shape(B.rows(), B.columns()));
  cholsolve(A.rows(), A.columns(), S.data(), S.stride(), A.data(), A.stride(),
      B.data(), B.stride());
  return A;
}

/**
 * Construct diagonal matrix. Diagonal elements are assigned to a given scalar
 * value, while all off-diagonal elements are assigned zero.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param a Scalar to assign to diagonal.
 * @param n Number of rows and columns.
 */
template<class T>
Array<T,2> diagonal(const Scalar<T>& a, const int n) {
  Array<T,2> B(make_shape(n, n));
  diagonal(a.data(), n, B.data(), B.stride());
  return B;
}

/**
 * Vector-vector dot product. Computes @f$x^\top y@f$, resulting in a scalar.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param x Vector.
 * @param y Vector.
 * 
 * @return Dot product.
 */
template<class T>
Scalar<T> dot(const Array<T,1>& x, const Array<T,1>& y) {
  assert(x.length() == y.length());
  Scalar<T> z;
  dot(x.length(), x.data(), x.stride(), y.data(), y.stride(), z.data());
  return z;
}

/**
 * Vector-matrix dot product. Equivalent to inner() with the arguments
 * reversed: computes @f$A^\top x@f$, resulting in a vector.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param x Vector.
 * @param A Matrix.
 * 
 * @return Vector giving the dot product of @p x with each column of @p A.
 */
template<class T>
Array<T,1> dot(const Array<T,1>& x, const Array<T,2>& A) {
  return inner(A, x);
}

/**
 * Equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double`, `float`, or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<bool,D> equal(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  equal(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Greater than comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double`, `float`, or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<bool,D> greater(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  greater(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Greater than or equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double`, `float`, or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<bool,D> greater_or_equal(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  greater_or_equal(A.width(), A.height(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Less than comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double`, `float`, or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<bool,D> less(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  less(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Less than or equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double`, `float`, or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<bool,D> less_or_equal(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  less_or_equal(A.width(), A.height(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Matrix-matrix Frobenius product. Computes @f$\langle A, B 
 * \rangle_\mathrm{F} = \mathrm{Tr}(A^\top B) = \sum_{ij} A_{ij} B_{ij}@f$,
 * resulting in a scalar.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * @param B Matrix.
 * 
 * @return Frobenius product.
 */
template<class T>
Scalar<T> frobenius(const Array<T,2>& A, const Array<T,2>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());
  Scalar<T> c;
  frobenius(A.rows(), A.columns(), A.data(), A.stride(), B.data(),
      B.stride(), c.data());
  return c;
}

/**
 * Matrix-vector inner product. Computes @f$y = A^\top x@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * @param x Vector.
 * 
 * @return Vector.
 */
template<class T>
Array<T,1> inner(const Array<T,2>& A, const Array<T,1>& x) {
  assert(A.rows() == x.length());
  
  Array<T,1> y(make_shape(A.columns()));
  inner(y.rows(), A.rows(), A.data(), A.stride(), x.data(), x.stride(),
      y.data(), y.stride());
  return y;
}

/**
 * Matrix-matrix inner product. Computes @f$C = A^\top B@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * @param B Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Array<T,2> inner(const Array<T,2>& A, const Array<T,2>& B) {
  assert(A.rows() == B.rows());

  Array<T,2> C(make_shape(A.columns(), B.columns()));
  inner(C.rows(), C.columns(), A.rows(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Logarithm of the beta function.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<T,D> lbeta(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<T,D> C(A.shape().compact());
  lbeta(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Logarithm of the binomial coefficient.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double`, `float` or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<T,D> lchoose(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<T,D> C(A.shape().compact());
  lchoose(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Logarithm of the multivariate gamma function.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<T,D> lgamma(const Array<T,D>& A, const Array<int,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<T,D> C(A.shape().compact());
  lgamma(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Logical `and`.
 * 
 * @ingroup array
 * 
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<int D>
Array<bool,D> logical_and(const Array<bool,D>& A, const Array<bool,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  logical_and(A.width(), A.height(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Logical `or`.
 * 
 * @ingroup array
 * 
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<int D>
Array<bool,D> logical_or(const Array<bool,D>& A, const Array<bool,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  logical_or(A.width(), A.height(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Not equal to comparison.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double`, `float`, or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<bool,D> not_equal(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<bool,D> C(A.shape().compact());
  not_equal(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Vector-vector outer product. Computes @f$A = xy^\top@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param x Vector.
 * @param y Vector.
 * 
 * @return Matrix.
 */
template<class T>
Array<T,2> outer(const Array<T,1>& x, const Array<T,1>& y) {
  Array<T,2> C(make_shape(x.length(), y.length()));
  outer(C.rows(), C.columns(), x.data(), x.stride(), y.data(), y.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Matrix-matrix outer product. Computes @f$C = AB^\top@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * @param B Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Array<T,2> outer(const Array<T,2>& A, const Array<T,2>& B) {
  assert(A.columns() == B.columns());

  Array<T,2> C(make_shape(A.rows(), B.rows()));
  outer(C.rows(), C.columns(), A.columns(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Power.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double`, `float`, or `int`).
 * @tparam U Element type (`double`, `float`, or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, int D, class U>
Array<T,D> pow(const Array<T,D>& A, const Array<U,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<T,D> C(A.shape().compact());
  pow(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Matrix-vector solve. Solves for @f$x@f$ in @f$Ax = y@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * @param y Vector.
 * 
 * @return Vector.
 */
template<class T>
Array<T,1> solve(const Array<T,2>& A, const Array<T,1>& y) {
  assert(A.rows() == A.columns());
  assert(A.rows() == y.length());

  Array<T,1> x(make_shape(y.length()));
  solve(x.rows(), A.data(), A.stride(), x.data(), x.stride(), y.data(),
      y.stride());
  return x;
}

/**
 * Matrix-matrix solve. Solves for @f$B@f$ in @f$AB = C@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * 
 * @param A Matrix.
 * @param C Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Array<T,2> solve(const Array<T,2>& A, const Array<T,2>& C) {
  assert(A.rows() == A.columns());
  assert(A.rows() == C.rows());

  Array<T,2> B(make_shape(C.rows(), C.columns()));
  solve(A.rows(), A.columns(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return A;
}

}
