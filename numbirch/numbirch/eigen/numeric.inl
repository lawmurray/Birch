/**
 * @file
 */
#pragma once

#include "numbirch/utility.hpp"
#include "numbirch/eigen/eigen.hpp"
#include "numbirch/numeric.hpp"

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
Array<T,2> chol(const Array<T,2>& S) {
  assert(rows(S) == columns(S));
  Array<T,2> L(shape(S));
  auto S1 = make_eigen(S);
  auto L1 = make_eigen(L);
  auto llt = S1.llt();
  if (llt.info() == Eigen::Success) {
    L1 = llt.matrixL();
  } else {
    L = T(0.0/0.0);
  }
  return L;
}

template<class T, class U, class>
Array<T,2> cholsolve(const Array<T,2>& L, const U& y) {
  assert(rows(L) == columns(L));
  Array<T,2> B(make_shape(rows(L), columns(L)));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto B1 = make_eigen(B);
  B1.noalias() = U1.solve(L1.solve(value(y)*B1.Identity(rows(B), columns(B))));
  return B;
}

template<class T, class>
Array<T,1> cholsolve(const Array<T,2>& L, const Array<T,1>& y) {
  assert(rows(L) == columns(L));
  assert(columns(L) == length(y));
  Array<T,1> x(shape(y));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  x1.noalias() = U1.solve(L1.solve(y1));
  return x;
}

template<class T, class>
Array<T,2> cholsolve(const Array<T,2>& L, const Array<T,2>& C) {
  assert(rows(L) == columns(L));
  assert(columns(L) == rows(C));
  Array<T,2> B(shape(C));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  B1.noalias() = U1.solve(L1.solve(C1));
  return B;
}

template<class T, class>
Array<T,0> dot(const Array<T,1>& x, const Array<T,1>& y) {
  assert(length(x) == length(y));
  Array<T,0> z;
  if (size(x) > 0) {
    auto x1 = make_eigen(x);
    auto y1 = make_eigen(y);
    z = x1.dot(y1);
  } else {
    z = T(0);
  }
  return z;
}

template<class T, class>
Array<T,0> frobenius(const Array<T,2>& x, const Array<T,2>& y) {
  assert(rows(x) == rows(y));
  assert(columns(x) == columns(y));
  Array<T,0> z;
  if (size(x) > 0 && size(y) > 0) {
    auto x1 = make_eigen(x);
    auto y1 = make_eigen(y);
    z = (x1.array()*y1.array()).sum();
  } else {
    z = T(0);
  }
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
Array<T,2> inv(const Array<T,2>& A) {
  assert(rows(A) == columns(A));
  Array<T,2> B(shape(A));
  auto A1 = make_eigen(A);
  auto B1 = make_eigen(B);
  B1.noalias() = A1.inverse();
  return B;
}

template<class T, class>
Array<T,0> ldet(const Array<T,2>& A) {
  if (size(A) == 0) {
    return T(0);
  } else {
    auto A1 = make_eigen(A);
    return A1.householderQr().logAbsDeterminant();
  }
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
Array<T,2> phi(const Array<T,2>& A) {
  Array<T,2> L(make_shape(rows(A), columns(A)));
  auto A1 = make_eigen(A).template triangularView<Eigen::Lower>();
  auto L1 = make_eigen(L);
  L1 = A1;
  L1.diagonal() *= 0.5;
  return L;
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
Array<T,2> tri(const Array<T,2>& A) {
  Array<T,2> L(make_shape(rows(A), columns(A)));
  auto A1 = make_eigen(A).template triangularView<Eigen::Lower>();
  auto L1 = make_eigen(L);
  L1 = A1;
  return L;
}

template<class T, class>
Array<T,1> triinner(const Array<T,2>& L, const Array<T,1>& x) {
  assert(rows(L) == length(x));
  Array<T,1> y(make_shape(columns(L)));
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  y1.noalias() = U1*x1;
  return y;
}

template<class T, class>
Array<T,2> triinner(const Array<T,2>& L, const Array<T,2>& B) {
  assert(rows(L) == rows(B));
  Array<T,2> C(make_shape(columns(L), columns(B)));
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  C1.noalias() = U1*B1;
  return C;
}

template<class T, class U, class>
Array<T,2> triinnersolve(const Array<T,2>& L, const U& y) {
  assert(rows(L) == columns(L));
  assert(columns(L) == length(y));
  Array<T,2> B(make_shape(rows(L), columns(L)));
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto B1 = make_eigen(B);
  B1.noalias() = U1.solve(value(y)*B1.Identity(rows(B), columns(B)));
  return B;

}

template<class T, class>
Array<T,1> triinnersolve(const Array<T,2>& L, const Array<T,1>& y) {
  assert(rows(L) == columns(L));
  assert(columns(L) == length(y));
  Array<T,1> x(shape(y));
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  x1.noalias() = U1.solve(y1);
  return x;
}

template<class T, class>
Array<T,2> triinnersolve(const Array<T,2>& L, const Array<T,2>& C) {
  assert(rows(L) == columns(L));
  assert(columns(L) == rows(C));
  Array<T,2> B(shape(C));
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  B1.noalias() = U1.solve(C1);
  return B;
}

template<class T, class>
Array<T,1> trimul(const Array<T,2>& L, const Array<T,1>& x) {
  assert(columns(L) == length(x));
  Array<T,1> y(make_shape(rows(L)));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  y1.noalias() = L1*x1;
  return y;
}

template<class T, class>
Array<T,2> trimul(const Array<T,2>& L, const Array<T,2>& B) {
  assert(columns(L) == rows(B));
  Array<T,2> C(make_shape(rows(L), columns(B)));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  C1.noalias() = L1*B1;
  return C;
}

template<class T, class>
Array<T,2> triouter(const Array<T,2>& A, const Array<T,2>& L) {
  assert(columns(A) == columns(L));
  Array<T,2> C(make_shape(rows(A), rows(L)));
  auto A1 = make_eigen(A);
  auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  auto C1 = make_eigen(C);
  C1.noalias() = A1*U1;
  return C;
}

template<class T, class U, class>
Array<T,2> trisolve(const Array<T,2>& L, const U& y) {
  assert(rows(L) == columns(L));
  Array<T,2> B(make_shape(rows(L), columns(L)));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto B1 = make_eigen(B);
  B1.noalias() = L1.solve(value(y)*B1.Identity(rows(B), columns(B)));
  return B;
}

template<class T, class>
Array<T,1> trisolve(const Array<T,2>& L, const Array<T,1>& y) {
  assert(rows(L) == columns(L));
  assert(columns(L) == length(y));
  Array<T,1> x(shape(y));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto x1 = make_eigen(x);
  auto y1 = make_eigen(y);
  x1.noalias() = L1.solve(y1);
  return x;
}

template<class T, class>
Array<T,2> trisolve(const Array<T,2>& L, const Array<T,2>& C) {
  assert(rows(L) == columns(L));
  assert(columns(L) == rows(C));
  Array<T,2> B(shape(C));
  auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  auto B1 = make_eigen(B);
  auto C1 = make_eigen(C);
  B1.noalias() = L1.solve(C1);
  return B;
}

}
