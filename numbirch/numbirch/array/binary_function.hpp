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
 * Lower-triangular Cholesky factor of a matrix multiplied by a vector.
 * Computes @f$y = Lx@f$, where @f$S = LL^\top@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * 
 * @param S Symmetric positive definite matrix.
 * @param x Vector.
 * 
 * @return Vector.
 */
template<class T>
Vector<T> cholmul(const Matrix<T>& S, const Vector<T>& x) {
  assert(S.rows() == S.columns());
  assert(S.columns() == x.length());

  Vector<T> y(make_shape(S.rows()));
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
 * @tparam T Value type.
 * 
 * @param S Symmetric positive definite matrix.
 * @param B Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Matrix<T> cholmul(const Matrix<T>& S, const Matrix<T>& B) {
  assert(S.rows() == S.columns());
  assert(S.columns() == S.rows());

  Matrix<T> C(make_shape(S.rows(), B.columns()));
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
 * @tparam T Value type.
 * 
 * @param A Matrix.
 * @param S Symmetric positive definite matrix.
 * 
 * @return Matrix.
 */
template<class T>
Matrix<T> cholouter(const Matrix<T>& A, const Matrix<T>& S) {
  assert(A.columns() == S.columns());
  assert(S.rows() == S.columns());

  Matrix<T> C(make_shape(A.rows(), S.rows()));
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
 * @tparam T Value type.
 * 
 * @param S Symmetric positive definite matrix.
 * @param y Vector.
 * 
 * @return Vector.
 */
template<class T>
Vector<T> cholsolve(const Matrix<T>& S, const Vector<T>& y) {
  assert(S.rows() == S.columns());
  assert(S.rows() == y.length());

  Vector<T> x(make_shape(y.length()));
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
 * @tparam T Value type.
 * 
 * @param S Symmetric positive definite matrix.
 * @param B Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Matrix<T> cholsolve(const Matrix<T>& S, const Matrix<T>& B) {
  assert(S.rows() == S.columns());
  assert(S.rows() == B.rows());

  Matrix<T> A(make_shape(B.rows(), B.columns()));
  cholsolve(A.rows(), A.columns(), S.data(), S.stride(), A.data(), A.stride(),
      B.data(), B.stride());
  return A;
}

/**
 * Copy sign of a number.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, class U, int D>
auto copysign(const Array<T,D>& A, const Array<U,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<T,D> C(A.shape().compact());
  copysign(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Copy sign of a number.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U,
    std::enable_if_t<std::is_arithmetic<U>::value,int> = 0>
auto copysign(const Scalar<T>& x, const U& y) {
  return copysign(x, Scalar<U>(y));
}

/**
 * Copy sign of a number.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U,
    std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
auto copysign(const T& x, const Scalar<U>& y) {
  return copysign(Scalar<T>(x), y);
}

/**
 * Construct diagonal matrix. Diagonal elements are assigned to a given scalar
 * value, while all off-diagonal elements are assigned zero.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * 
 * @param x Scalar to assign to diagonal.
 * @param n Number of rows and columns.
 */
template<class T>
Matrix<T> diagonal(const Scalar<T>& x, const int n) {
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
 * @tparam T Value type.
 * 
 * @param x Scalar to assign to diagonal.
 * @param n Number of rows and columns.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
Matrix<T> diagonal(const T& x, const int n) {
  return diagonal(Scalar<T>(x), n);
}

/**
 * Vector-vector dot product. Computes @f$x^\top y@f$, resulting in a scalar.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param x Vector.
 * @param y Vector.
 * 
 * @return Dot product.
 */
template<class T>
Scalar<T> dot(const Vector<T>& x, const Vector<T>& y) {
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
 * @tparam T Value type.
 * 
 * @param x Vector.
 * @param A Matrix.
 * 
 * @return Vector giving the dot product of @p x with each column of @p A.
 */
template<class T>
Vector<T> dot(const Vector<T>& x, const Matrix<T>& A) {
  return inner(A, x);
}

/**
 * Matrix-matrix Frobenius product. Computes @f$\langle A, B 
 * \rangle_\mathrm{F} = \mathrm{Tr}(A^\top B) = \sum_{ij} A_{ij} B_{ij}@f$,
 * resulting in a scalar.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * 
 * @param A Matrix.
 * @param B Matrix.
 * 
 * @return Frobenius product.
 */
template<class T>
Scalar<T> frobenius(const Matrix<T>& A, const Matrix<T>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());
  Scalar<T> c;
  frobenius(A.rows(), A.columns(), A.data(), A.stride(), B.data(),
      B.stride(), c.data());
  return c;
}

/**
 * Hadamard (element-wise) matrix multiplication.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, class U, int D>
auto hadamard(const Array<T,D>& A, const Array<U,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<decltype(T()*U()),D> C(A.shape().compact());
  hadamard(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Hadamard (element-wise) matrix multiplication.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U,
    std::enable_if_t<std::is_arithmetic<U>::value,int> = 0>
auto hadamard(const Scalar<T>& x, const U& y) {
  return hadamard(x, Scalar<U>(y));
}

/**
 * Hadamard (element-wise) matrix multiplication.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U,
    std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
auto hadamard(const T& x, const Scalar<U>& y) {
  return hadamard(Scalar<T>(x), y);
}

/**
 * Matrix-vector inner product. Computes @f$y = A^\top x@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * 
 * @param A Matrix.
 * @param x Vector.
 * 
 * @return Vector.
 */
template<class T>
Vector<T> inner(const Matrix<T>& A, const Vector<T>& x) {
  assert(A.rows() == x.length());
  
  Vector<T> y(make_shape(A.columns()));
  inner(y.rows(), A.rows(), A.data(), A.stride(), x.data(), x.stride(),
      y.data(), y.stride());
  return y;
}

/**
 * Matrix-matrix inner product. Computes @f$C = A^\top B@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * 
 * @param A Matrix.
 * @param B Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Matrix<T> inner(const Matrix<T>& A, const Matrix<T>& B) {
  assert(A.rows() == B.rows());

  Matrix<T> C(make_shape(A.columns(), B.columns()));
  inner(C.rows(), C.columns(), A.rows(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Logarithm of the beta function.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, class U, int D>
auto lbeta(const Array<T,D>& A, const Array<U,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<decltype(lbeta(T(), U())),D> C(A.shape().compact());
  lbeta(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Logarithm of the beta function.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U,
    std::enable_if_t<std::is_arithmetic<U>::value,int> = 0>
auto lbeta(const Scalar<T>& x, const U& y) {
  return lbeta(x, Scalar<U>(y));
}

/**
 * Logarithm of the beta function.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U,
    std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
auto lbeta(const T& x, const Scalar<U>& y) {
  return lbeta(Scalar<T>(x), y);
}

/**
 * Logarithm of the binomial coefficient.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, class U, int D>
auto lchoose(const Array<T,D>& A, const Array<U,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<decltype(lchoose(T(), U())),D> C(A.shape().compact());
  lchoose(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Logarithm of the binomial coefficient.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U,
    std::enable_if_t<std::is_arithmetic<U>::value,int> = 0>
auto lchoose(const Scalar<T>& x, const U& y) {
  return lchoose(x, Scalar<U>(y));
}

/**
 * Logarithm of the binomial coefficient.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U,
    std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
auto lchoose(const T& x, const Scalar<U>& y) {
  return lchoose(Scalar<T>(x), y);
}

/**
 * Logarithm of the multivariate gamma function.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
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
 * Logarithm of the multivariate gamma function.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T>
auto lgamma(const Scalar<T>& x, const int& y) {
  return lgamma(x, Scalar<int>(y));
}

/**
 * Logarithm of the multivariate gamma function.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T>
auto lgamma(const T& x, const Scalar<int>& y) {
  return lgamma(Scalar<T>(x), y);
}

/**
 * Vector-vector outer product. Computes @f$A = xy^\top@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * 
 * @param x Vector.
 * @param y Vector.
 * 
 * @return Matrix.
 */
template<class T>
Matrix<T> outer(const Vector<T>& x, const Vector<T>& y) {
  Matrix<T> C(make_shape(x.length(), y.length()));
  outer(C.rows(), C.columns(), x.data(), x.stride(), y.data(), y.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Matrix-matrix outer product. Computes @f$C = AB^\top@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * 
 * @param A Matrix.
 * @param B Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Matrix<T> outer(const Matrix<T>& A, const Matrix<T>& B) {
  assert(A.columns() == B.columns());

  Matrix<T> C(make_shape(A.rows(), B.rows()));
  outer(C.rows(), C.columns(), A.columns(), A.data(), A.stride(), B.data(),
      B.stride(), C.data(), C.stride());
  return C;
}

/**
 * Power.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * @tparam D Number of dimensions.
 * 
 * @param A %Array.
 * @param B %Array.
 * 
 * @return %Array.
 */
template<class T, class U, int D>
Array<T,D> pow(const Array<T,D>& A, const Array<U,D>& B) {
  assert(A.rows() == B.rows());
  assert(A.columns() == B.columns());

  Array<decltype(T()*U()),D> C(A.shape().compact());
  pow(A.width(), A.height(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return C;
}

/**
 * Power.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U,
    std::enable_if_t<std::is_arithmetic<U>::value,int> = 0>
auto pow(const Scalar<T>& x, const U& y) {
  return pow(x, Scalar<U>(y));
}

/**
 * Power.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam U Value type.
 * 
 * @param x %Scalar.
 * @param y %Scalar.
 * 
 * @return %Scalar.
 */
template<class T, class U,
    std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
auto pow(const T& x, const Scalar<U>& y) {
  return pow(Scalar<T>(x), y);
}

/**
 * Matrix-vector solve. Solves for @f$x@f$ in @f$Ax = y@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * 
 * @param A Matrix.
 * @param y Vector.
 * 
 * @return Vector.
 */
template<class T>
Vector<T> solve(const Matrix<T>& A, const Vector<T>& y) {
  assert(A.rows() == A.columns());
  assert(A.rows() == y.length());

  Vector<T> x(make_shape(y.length()));
  solve(x.rows(), A.data(), A.stride(), x.data(), x.stride(), y.data(),
      y.stride());
  return x;
}

/**
 * Matrix-matrix solve. Solves for @f$B@f$ in @f$AB = C@f$.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * 
 * @param A Matrix.
 * @param C Matrix.
 * 
 * @return Matrix.
 */
template<class T>
Matrix<T> solve(const Matrix<T>& A, const Matrix<T>& C) {
  assert(A.rows() == A.columns());
  assert(A.rows() == C.rows());

  Matrix<T> B(make_shape(C.rows(), C.columns()));
  solve(A.rows(), A.columns(), A.data(), A.stride(), B.data(), B.stride(),
      C.data(), C.stride());
  return A;
}

}
