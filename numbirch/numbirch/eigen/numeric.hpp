/**
 * @file
 */
#pragma once

#include "numbirch/eigen/eigen.hpp"
#include "numbirch/common/functor.hpp"

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
  auto llt = S1.llt();
  if (llt.info() == Eigen::Success) {
    B1.noalias() = llt.solve(B1.Identity(rows(B), columns(B)));
  } else {
    /* try again with a more stable factorization */
    auto qr = S1.fullPivHouseholderQr();
    B1.noalias() = qr.solve(B1.Identity(rows(B), columns(B)));
  }
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
  auto llt = S1.llt();
  if (llt.info() == Eigen::Success) {
    /* we have $S = LL^\top$; the determinant of $S$ is the square of the
     * determinant of $L$, the determinant of $L$ is the product of the
     * elements along the diagonal, as it is a triangular matrix; adjust for
     * log-determinant */
    return 2.0*llt.matrixLLT().diagonal().array().log().sum();
  } else {
    /* try again with a more stable factorization */
    return S1.fullPivHouseholderQr().logAbsDeterminant();
  }
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
  auto llt = S1.llt();
  if (llt.info() == Eigen::Success) {
    y1.noalias() = llt.matrixL()*x1;
  } else if constexpr (std::is_same_v<T,float>) {
    /* try again in double precision */
    auto S2 = S1.template cast<double>();
    auto x2 = x1.template cast<double>();
    auto llt = S2.llt();
    y1.noalias() = (llt.matrixL()*x2).template cast<T>();
  } else {
    assert(llt.info() == Eigen::Success);
  }
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
  auto llt = S1.llt();
  if (llt.info() == Eigen::Success) {
    C1.noalias() = llt.matrixL()*B1;
  } else if constexpr (std::is_same_v<T,float>) {
    /* try again in double precision */
    auto S2 = S1.template cast<double>();
    auto B2 = B1.template cast<double>();
    auto llt = S2.llt();
    C1.noalias() = (llt.matrixL()*B2).template cast<T>();
  } else {
    assert(llt.info() == Eigen::Success);
  }
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
  auto llt = S1.llt();
  if (llt.info() == Eigen::Success) {
    C1.noalias() = A1*llt.matrixU();
  } else if constexpr (std::is_same_v<T,float>) {
    /* try again in double precision */
    auto A2 = A1.template cast<double>();
    auto S2 = S1.template cast<double>();
    auto llt = S2.llt();
    C1.noalias() = (A2*llt.matrixU()).template cast<T>();
  } else {
    assert(llt.info() == Eigen::Success);
  }
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
  auto llt = S1.llt();
  if (llt.info() == Eigen::Success) {
    x1.noalias() = llt.solve(y1);
  } else {
    x1.noalias() = S1.fullPivHouseholderQr().solve(y1);
  }
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
  auto llt = S1.llt();
  if (llt.info() == Eigen::Success) {
    B1.noalias() = llt.solve(C1);
  } else {
    /* try again with a more stable factorization */
    B1.noalias() = S1.fullPivHouseholderQr().solve(C1);
  }
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
