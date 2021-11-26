/**
 * @file
 */
#pragma once

#include "numbirch/eigen/eigen.hpp"
#include "numbirch/common/functor.hpp"

#include <iostream>

namespace numbirch {

template<class T, class>
Array<T,1> operator*(const Array<T,2>& A, const Array<T,1>& x) {
  assert(columns(A) == length(x));
  Array<T,1> y(make_shape(rows(A)));
  auto A1 = make_eigen(A);
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  y1.noalias() = A1*x1;
  return y;
}

template<class T, class>
Array<T,2> operator*(const Array<T,2>& A, const Array<T,2>& B) {
  assert(columns(A) == rows(B));
  Array<T,2> C(make_shape(rows(A), columns(B)));
  auto A1 = make_eigen(A);
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  C1.noalias() = A1*B1;
  return C;
}

template<class T, class>
Array<T,2> cholinv(const Array<T,2>& S) {
  assert(rows(S) == columns(S));
  Array<T,2> B(shape(S));
  auto S1 = make_eigen(S);
  auto B1 = make_eigen(B);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  B1.noalias() = ldlt.solve(B1.Identity(rows(B), columns(B)));
  return B;
}

template<class R, class T, class>
Array<R,0> count(const T& x) {
  return make_eigen(x).unaryExpr(count_functor<R>()).sum();
}

template<class R, class T, class>
Array<R,2> diagonal(const T& x, const int n) {
  Array<R,2> B(make_shape(n, n));
  auto B1 = make_eigen(B);
  B1.noalias() = R(element(data(x)))*B1.Identity(n, n);
  return B;
}

template<class T, class>
Array<T,2> inv(const Array<T,2>& A) {
  assert(rows(A) == columns(A));
  Array<T,2> B(shape(A));
  auto A1 = make_eigen(A);
  auto B1 = make_eigen(B);
  B1.noalias() = A1.inverse();
  return B;
}

template<class T, class>
Array<T,0> lcholdet(const Array<T,2>& S) {
  auto S1 = make_eigen(S);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);

  /* Eigen's LDLT decomposition factorizes as $S = P^\top LDL^\top P$; the $L$
   * matrix has unit diagonal and thus determinant 1, the $P$ matrix is a
   * permutation matrix and thus has determinant $\pm 1$, which squares to 1,
   * leaving only the $D$ matrix with its determinant being the product along
   * the main diagonal (log and sum) */
  return ldlt.vectorD().array().log().sum();
}

template<class T, class>
Array<T,0> ldet(const Array<T,2>& A) {
  auto A1 = make_eigen(A);
  return A1.householderQr().logAbsDeterminant();
}

template<class R, class T, class>
Array<R,1> single(const T& i, const int n) {
  Array<R,1> x(make_shape(n));
  auto x1 = make_eigen(x);
  x1.noalias() = x1.Zero(n, 1);
  x1(element(data(i)) - 1) = R(1);
  return x;
}

template<class R, class T, class U, class>
Array<R,2> single(const T& i, const U& j, const int m, const int n) {
  Array<R,2> x(make_shape(m, n));
  auto x1 = make_eigen(x);
  x1.noalias() = x1.Zero(m, n);
  x1(element(data(i)) - 1, element(data(j)) - 1) = R(1);
  return x;
}

template<class R, class T, class>
Array<R,0> sum(const T& x) {
  return make_eigen(x).sum();
}

template<class T, class>
Array<T,0> trace(const Array<T,2>& A) {
  return make_eigen(A).trace();
}

template<class T, class>
Array<T,2> transpose(const Array<T,2>& A) {
  Array<T,2> B(make_shape(columns(A), rows(A)));
  auto A1 = make_eigen(A);
  auto B1 = make_eigen(B);
  B1.noalias() = A1.transpose();
  return B;
}

template<class T, class>
Array<T,1> cholmul(const Array<T,2>& S, const Array<T,1>& x) {
  assert(rows(S) == columns(S));
  assert(columns(S) == length(x));
  Array<T,1> y(make_shape(rows(S)));
  auto S1 = make_eigen(S);
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  y1.noalias() = ldlt.transpositionsP().transpose()*(ldlt.matrixL()*
      (ldlt.vectorD().cwiseMax(0.0).cwiseSqrt().cwiseProduct(x1)));
  return y;
}

template<class T, class>
Array<T,2> cholmul(const Array<T,2>& S, const Array<T,2>& B) {
  assert(rows(S) == columns(S));
  assert(columns(S) == rows(B));
  Array<T,2> C(shape(B));
  auto S1 = make_eigen(S);
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  C1.noalias() = ldlt.transpositionsP().transpose()*(ldlt.matrixL()*
      (ldlt.vectorD().cwiseMax(0.0).cwiseSqrt().asDiagonal()*B1));
  return C;
}

template<class T, class>
Array<T,2> cholouter(const Array<T,2>& A, const Array<T,2>& S) {
  assert(columns(A) == columns(S));
  assert(rows(S) == columns(S));
  Array<T,2> C(make_shape(rows(A), rows(S)));
  auto A1 = make_eigen(A);
  auto S1 = make_eigen(S);
  auto C1 = make_eigen(C);
  auto ldlt = S1.ldlt();
  if (ldlt.info() != Eigen::Success) {
    std::cerr << "-----------------------------------------" << std::endl;
    std::cerr << S1 << std::endl;
  }
  assert(ldlt.info() == Eigen::Success);
  C1.noalias() = (ldlt.transpositionsP().transpose()*(ldlt.matrixL()*
      (ldlt.vectorD().cwiseMax(0.0).cwiseSqrt().asDiagonal()*
      A1.transpose()))).transpose();
  return C;
}

template<class T, class>
Array<T,1> cholsolve(const Array<T,2>& S, const Array<T,1>& y) {
  assert(rows(S) == columns(S));
  assert(columns(S) == length(y));
  Array<T,1> x(shape(y));
  auto S1 = make_eigen(S);
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  x1.noalias() = ldlt.solve(y1);
  return x;
}

template<class T, class>
Array<T,2> cholsolve(const Array<T,2>& S, const Array<T,2>& C) {
  assert(rows(S) == columns(S));
  assert(columns(S) == rows(C));
  Array<T,2> B(shape(C));
  auto S1 = make_eigen(S);
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  B1.noalias() = ldlt.solve(C1);
  return B;
}

template<class T, class>
Array<T,0> dot(const Array<T,1>& x, const Array<T,1>& y) {
  assert(length(x) == length(y));
  Array<T,0> z;
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  z = x1.dot(y1);
  return z;
}

template<class T, class>
Array<T,0> frobenius(const Array<T,2>& x, const Array<T,2>& y) {
  Array<T,0> z;
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  z = (x1.array()*y1.array()).sum();
  return z;
}

template<class T, class>
Array<T,1> inner(const Array<T,2>& A, const Array<T,1>& x) {
  assert(rows(A) == length(x));
  Array<T,1> y(make_shape(columns(A)));
  auto A1 = make_eigen(A);
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  y1.noalias() = A1.transpose()*x1;
  return y;
}

template<class T, class>
Array<T,2> inner(const Array<T,2>& A, const Array<T,2>& B) {
  assert(rows(A) == rows(B));
  Array<T,2> C(make_shape(columns(A), columns(B)));
  auto A1 = make_eigen(A);
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  C1.noalias() = A1.transpose()*B1;
  return C;
}

template<class T, class>
Array<T,2> outer(const Array<T,1>& x, const Array<T,1>& y) {
  Array<T,2> A(make_shape(length(x), length(y)));
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  auto A1 = make_eigen(A);
  A1.noalias() = x1*y1.transpose();
  return A;
}

template<class T, class>
Array<T,2> outer(const Array<T,2>& A, const Array<T,2>& B) {
  assert(columns(A) == columns(B));
  Array<T,2> C(make_shape(rows(A), rows(B)));
  auto A1 = make_eigen(A);
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  C1.noalias() = A1*B1.transpose();
  return C;
}

template<class T, class>
Array<T,1> solve(const Array<T,2>& A, const Array<T,1>& y) {
  assert(rows(A) == columns(A));
  assert(columns(A) == length(y));
  Array<T,1> x(shape(y));
  auto A1 = make_eigen(A);
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  x1.noalias() = A1.householderQr().solve(y1);
  return x;
}

template<class T, class>
Array<T,2> solve(const Array<T,2>& A, const Array<T,2>& C) {
  assert(rows(A) == columns(A));
  assert(columns(A) == rows(C));
  Array<T,2> B(shape(C));
  auto A1 = make_eigen(A);
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  B1.noalias() = A1.householderQr().solve(C1);
  return B;
}

}
