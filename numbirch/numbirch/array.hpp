/**
 * @file
 * 
 * NumBirch interface.
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
 * Negation.
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
Array<T,D> operator-(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  neg(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Addition.
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
Array<T,D> operator+(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(B.columns() == B.columns());

  Array<T,D> C(A.shape().compact());
  add(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Scalar addition.
 * 
 * @ingroup array
 * 
 * @tparam T Scalar type (`double`, `float` or `int`).
 * @tparam U Scalar type (`double`, `float` or `int`).
 * 
 * @param a Scalar.
 * @param b Scalar.
 * 
 * @return Scalar.
 */
template<class T, class U,
    std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
Scalar<U> operator+(const T a, const Scalar<U>& b) {
  Scalar<U> c;
  add(1, 1, U(a), b.data(), b.stride(), c.data(), c.stride());
  return c;
}

/**
 * Scalar addition.
 * 
 * @ingroup array
 * 
 * @tparam T Scalar type (`double`, `float` or `int`).
 * @tparam U Scalar type (`double`, `float` or `int`).
 * 
 * @param a Scalar.
 * @param b Scalar.
 * 
 * @return Scalar.
 */
template<class T, class U,
    std::enable_if_t<std::is_arithmetic<U>::value,int> = 0>
Scalar<T> operator+(const Scalar<T>& a, const U b) {
  return b + a;
}

/**
 * Subtraction.
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
Array<T,D> operator-(const Array<T,D>& A, const Array<T,D>& B) {
  assert(A.rows() == B.rows());
  assert(B.columns() == B.columns());

  Array<T,D> C(A.shape().compact());
  sub(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Scalar subtraction.
 * 
 * @ingroup array
 * 
 * @tparam T Scalar type (`double`, `float` or `int`).
 * @tparam U Scalar type (`double`, `float` or `int`).
 * 
 * @param a Scalar.
 * @param b Scalar.
 * 
 * @return %Array.
 */
template<class T, class U,
    std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
Scalar<U> operator-(const T a, const Scalar<U>& b) {
  return -a + b;
}

/**
 * Scalar subtraction.
 * 
 * @ingroup array
 * 
 * @tparam T Scalar type (`double`, `float` or `int`).
 * @tparam U Scalar type (`double`, `float` or `int`).
 * 
 * @param a Scalar.
 * @param b Scalar.
 * 
 * @return %Array.
 */
template<class T, class U,
    std::enable_if_t<std::is_arithmetic<U>::value,int> = 0>
Scalar<T> operator-(const Scalar<T>& a, const U b) {
  Scalar<T> c;
  sub(1, 1, a.data(), a.stride(), b, c.data(), c.stride());
  return c;
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
Array<T,D> rectify(const Array<T,D>& A) {
  Array<T,D> B(A.shape().compact());
  rectify(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride());
  return B;
}

/**
 * Linear combination of matrices.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param a Coefficient on `A`.
 * @param A %Array.
 * @param b Coefficient on `B`.
 * @param B %Array.
 * @param c Coefficient on `C`.
 * @param C %Array.
 * @param e Coefficient on `D`.
 * @param E %Array.
 * 
 * @return %Array.
 */
template<class T, int D>
Array<T,D> combine(const T a, const Array<T,D>& A, const T b,
    const Array<T,D>& B, const T c, const Array<T,D>& C, const T e,
    const Array<T,D>& E) {
  assert(A.rows() == B.rows());
  assert(A.rows() == C.rows());
  assert(A.rows() == E.rows());
  assert(A.columns() == B.columns());
  assert(A.columns() == C.columns());
  assert(A.columns() == E.columns());

  Array<T,D> F(A.shape().compact());
  combine(F.width(), F.height(), a, A.data(), A.stride(), b, B.data(),
      B.stride(), c, C.data(), C.stride(), e, E.data(), E.stride(), F.data(),
      F.stride());
  return F;
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
 * Scalar division.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam U Scalar type (`double`, `float` or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param b Scalar.
 * 
 * @return %Array.
 */
template<class T, int D, class U,
    std::enable_if_t<std::is_arithmetic<U>::value,int> = 0>
Array<T,D> operator/(const Array<T,D>& A, const U b) {
  Array<T,D> C(A.shape().compact());
  div(A.width(), A.height(), A.data(), A.stride(), b, C.data(), C.stride());
  return C;
}

/**
 * Scalar division.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam U Scalar type (`double`, `float` or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param b Scalar.
 * 
 * @return %Array.
 */
template<class T, int D, class U>
Array<T,D> operator/(const Array<T,D>& A, const Scalar<U>& b) {
  Array<T,D> C(A.shape().compact());
  div(A.width(), A.height(), A.data(), A.stride(), b.data(), C.data(),
      C.stride());
  return C;
}

/**
 * Scalar multiplication.
 * 
 * @ingroup array
 * 
 * @tparam T Scalar type (`double`, `float` or `int`).
 * @tparam U Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param a Scalar.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, class U, int D,
    std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
Array<U,D> operator*(const T a, const Array<U,D>& B) {
  Array<U,D> C(B.shape().compact());
  mul(C.width(), C.height(), a, B.data(), B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Scalar multiplication.
 * 
 * @ingroup array
 * 
 * @tparam T Scalar type (`double`, `float` or `int`).
 * @tparam U Element type (`double` or `float`).
 * @tparam D Number of dimensions.
 * 
 * @param a Scalar.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, class U, int D>
Array<U,D> operator*(const Scalar<T>& a, const Array<U,D>& B) {
  Array<U,D> C(B.shape().compact());
  mul(C.width(), C.height(), a.data(), B.data(), B.stride(), C.data(),
      C.stride());
  return C;
}

/**
 * Scalar multiplication.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam U Scalar type (`double`, `float` or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param b %Array.
 * 
 * @return %Array.
 */
template<class T, int D, class U,
    std::enable_if_t<std::is_arithmetic<U>::value,int> = 0>
Array<T,D> operator*(const Array<T,D>& A, const U b) {
  return b*A;
}

/**
 * Scalar multiplication.
 * 
 * @ingroup array
 * 
 * @tparam T Element type (`double` or `float`).
 * @tparam U Scalar type (`double`, `float` or `int`).
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param b %Array.
 * 
 * @return %Array.
 */
template<class T, int D, class U>
Array<T,D> operator*(const Array<T,D>& A, const Scalar<U>& b) {
  return b*A;
}

/**
 * Matrix-vector multiplication. Computes @f$y = Ax@f$.
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
Array<T,1> operator*(const Array<T,2>& A, const Array<T,1>& x) {
  assert(A.columns() == x.length());

  Array<T,1> y(make_shape(A.rows()));
  mul(A.rows(), A.columns(), A.data(), A.stride(), x.data(), x.stride(),
      y.data(), y.stride());
  return y;
}

/**
 * Matrix-matrix multiplication. Computes @f$C = AB@f$.
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
Array<T,2> operator*(const Array<T,2>& A, const Array<T,2>& B) {
  assert(A.columns() == B.rows());

  Array<T,2> C(make_shape(A.rows(), B.columns()));
  mul(C.rows(), C.columns(), A.columns(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
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
 * Matrix sum of elements.
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
Array<T,2> inv(const Array<T,2>& A) {
  Array<T,2> B(make_shape(A.rows(), A.columns()));
  inv(A.rows(), A.data(), A.stride(), B.data(), B.stride());
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
Array<T,2> cholinv(const Array<T,2>& S) {
  Array<T,2> B(make_shape(S.rows(), S.columns()));
  cholinv(S.rows(), S.data(), S.stride(), B.data(), B.stride());
  return B;
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
Scalar<T> ldet(const Array<T,2>& A) {
  Scalar<T> b;
  ldet(A.rows(), A.data(), A.stride(), b.data());
  return b;
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
Scalar<T> lcholdet(const Array<T,2>& S) {
  Scalar<T> b;
  lcholdet(S.rows(), S.data(), S.stride(), b.data());
  return b;
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
Array<T,2> diagonal(const T a, const int n) {
  Array<T,2> B(make_shape(n, n));
  diagonal(a, n, B.data(), B.stride());
  return B;
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
Array<T,2> diagonal(const T* a, const int n) {
  Array<T,2> B(make_shape(n, n));
  diagonal(a, n, B.data(), B.stride());
  return B;
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
Array<T,2> transpose(const Array<T,2>& A) {
  Array<T,2> B(make_shape(A.columns(), A.rows()));
  transpose(B.rows(), B.columns(), A.data(), A.stride(), B.data(),
      B.stride());
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
Scalar<T> trace(const Array<T,2>& A) {
  Scalar<T> b;
  trace(A.rows(), A.columns(), A.data(), A.stride(), b.data());
  return b;
}

}
