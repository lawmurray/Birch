/**
 * @file
 */
#pragma once

#include "numbirch/eigen/eigen.hpp"

namespace numbirch {

template<class R, class T, class>
convert_t<R,T> operator+(const T& x) {
  if constexpr (std::is_same_v<R,T>) {
    return x;
  } else {
    convert_t<R,T> y(x.shape());
    auto x1 = make_eigen_matrix(x);
    auto y1 = make_eigen_matrix(y);
    y1.noalias() = x1.unaryExpr(identity_functor<R>());
    return y1;
  }
}

template<class R, class T, class>
convert_t<R,T> operator-(const T& x) {
  convert_t<R,T> y(x.shape());
  auto x1 = make_eigen_matrix(x);
  auto y1 = make_eigen_matrix(y);
  y1.noalias() = x1.unaryExpr(negate_functor<R>());
  return y1;
}

template<class T, class U, class>
convert_t<R,T,U> operator+(const T& x, const U& y) {
  assert(conforms(x, y));
  convert_t<R,T,U> y(x.shape());
  auto A1 = make_eigen_matrix(x).template cast<R>();
  auto B1 = make_eigen_matrix(y).template cast<R>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1 + B1;
}

template<class T, class U, class>
void div(const T& x, const U* b,
    V* C, const int ldC) {
  auto A1 = make_eigen_matrix(x).template cast<V>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1/(*b);
}

template<class T>
void equal(const T& x, const U& y, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, equal_functor<T>());
}

template<class T>
void greater(const T& x, const U& y, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, greater_functor<T>());
}

template<class T>
void greater_or_equal(const T& x,
    const T* B, const int ldB, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, greater_or_equal_functor<T>());
}

template<class T>
void less(const T& x, const U& y, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, less_functor<T>());
}

template<class T>
void less_or_equal(const T& x,
    const T* B, const int ldB, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, less_or_equal_functor<T>());
}

template<class T>
void neg(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.noalias() = -A1;
}

void logical_and(const T& x,
    const bool* B, const int ldB, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, logical_and_functor());
}

void logical_or(const T& x,
    const bool* B, const int ldB, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, logical_or_functor());
}

template<class T, class U, class>
void mul(const T* a, const U* B, const int ldB,
    V* C, const int ldC) {
  auto B1 = make_eigen_matrix(b).template cast<V>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = (*a)*B1;
}

template<class T>
void mul(const T& x, const T* x,
    const int incx, T* y, const int incy) {
  auto A1 = make_eigen_matrix(x);
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, m, incy);
  y1.noalias() = A1*x1;
}

template<class T>
void mul(const int k, const T& x,
    const T* B, const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, k, ldA);
  auto B1 = make_eigen_matrix(B, k, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1*B1;
}

template<class T>
void not_equal(const T& x,
    const T* B, const int ldB, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, not_equal_functor<T>());
}

template<class T, class U, class>
void sub(const T& x, const U& y) {
  auto A1 = make_eigen_matrix(x).template cast<V>();
  auto B1 = make_eigen_matrix(b).template cast<V>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1 - B1;
}

template<class T>
void abs(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().abs();
}

template<class T>
void acos(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().acos();
}

template<class T>
void asin(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().asin();
}

template<class T>
void atan(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().atan();
}

template<class T>
void ceil(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().ceil();
}

template<class T>
void cholinv(const int n, const T* S, const int ldS, T* B, const int ldB) {
  auto S1 = make_eigen_matrix(S, n, n, ldS);
  auto B1 = make_eigen_matrix(B, n, n, ldB);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  B1.noalias() = ldlt.solve(B1.Identity(n, n));
}

template<class T>
void cos(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().cos();
}

template<class T>
void cosh(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().cosh();
}

template<class T>
void count(const T& x, int* b) {
  auto A1 = make_eigen_matrix(x);
  *b = A1.unaryExpr(count_functor<T>()).sum();
}

template<class T>
void diagonal(const T* a, const int n, T* B, const int ldB) {
  auto B1 = make_eigen_matrix(B, n, n, ldB);
  B1.noalias() = (*a)*B1.Identity(n, n);
}

template<class T>
void digamma(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.noalias() = A1.unaryExpr(digamma_functor<T>());
}

template<class T>
void exp(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().exp();
}

template<class T>
void expm1(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.noalias() = A1.unaryExpr(expm1_functor<T>());
}

template<class T>
void floor(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().floor();
}

template<class T>
void inv(const int n, const T& x, T* B, const int ldB) {
  auto A1 = make_eigen_matrix(A, n, n, ldA);
  auto B1 = make_eigen_matrix(B, n, n, ldB);
  B1.noalias() = A1.inverse();
}

template<class T>
void lcholdet(const int n, const T* S, const int ldS, T* b) {
  auto S1 = make_eigen_matrix(S, n, n, ldS);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);

  /* Eigen's LDLT decomposition factorizes as $S = P^\top LDL^\top P$; the $L$
   * matrix has unit diagonal and thus determinant 1, the $P$ matrix is a
   * permutation matrix and thus has determinant $\pm 1$, which squares to 1,
   * leaving only the $D$ matrix with its determinant being the product along
   * the main diagonal (log and sum) */
  *b = ldlt.vectorD().array().log().sum();
}

template<class T>
void ldet(const int n, const T& x, T* b) {
  auto A1 = make_eigen_matrix(A, n, n, ldA);
  *b = A1.householderQr().logAbsDeterminant();
}

template<class T>
void lfact(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.noalias() = A1.unaryExpr(lfact_functor<T>());
}

template<class T>
void lfact_grad(const T* G, const int ldG,
    const T& x, T* B, const int ldB) {
  auto G1 = make_eigen_matrix(G, m, n, ldG);
  auto A1 = make_eigen_matrix(x).template cast<T>();
  auto B1 = make_eigen_matrix(b);
  B1.noalias() = G1.binaryExpr(A1, lfact_grad_functor<T>());
}

template<class T>
void lgamma(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.noalias() = A1.unaryExpr(lgamma_functor<T>());
}

template<class T>
void log(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().log();
}

template<class T>
void log1p(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().log1p();
}

template<class T>
void rcp(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.noalias() = A1.unaryExpr(rcp_functor<T>());
}

template<class T>
void rectify(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().max(T(0));
}

template<class T>
void rectify_grad(const T* G, const int ldG,
    const T& x, T* B, const int ldB) {
  auto G1 = make_eigen_matrix(G, m, n, ldG);
  auto A1 = make_eigen_matrix(x).template cast<T>();
  auto B1 = make_eigen_matrix(b);
  B1.noalias() = G1.binaryExpr(A1, rectify_grad_functor<T>());
}

template<class T>
void round(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().round();
}

template<class T>
void sin(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().sin();
}

template<class T>
void single(const int* i, const int n, T* x, const int incx) {
  auto x1 = make_eigen_vector(x, n, incx);
  x1.noalias() = x1.Zero(n, 1);
  x1(*i - 1) = T(1);
}

template<class T>
void sinh(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().sinh();
}

template<class T>
void sqrt(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().sqrt();
}

template<class T>
void sum(const T& x, T* b) {
  auto A1 = make_eigen_matrix(x);
  *b = A1.sum();
}

template<class T>
void tan(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().tan();
}

template<class T>
void tanh(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  B1.array() = A1.array().tanh();
}

template<class T>
void trace(const T& x, T* b) {
  auto A1 = make_eigen_matrix(x);
  *b = A1.trace();
}

template<class T>
void transpose(const T& x, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, n, m, ldA);
  auto B1 = make_eigen_matrix(b);
  B1.noalias() = A1.transpose();
}

template<class T>
void cholmul(const int n, const T* S, const int ldS, const T* x,
    const int incx, T* y, const int incy) {
  auto S1 = make_eigen_matrix(S, n, n, ldS);
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  y1.noalias() = ldlt.transpositionsP().transpose()*(ldlt.matrixL()*
      (ldlt.vectorD().cwiseMax(0.0).cwiseSqrt().cwiseProduct(x1)));
}

template<class T>
void cholmul(const T* S, const int ldS, const U& y, T* C, const int ldC) {
  auto S1 = make_eigen_matrix(S, m, m, ldS);
  auto B1 = make_eigen_matrix(b);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  C1.noalias() = ldlt.transpositionsP().transpose()*(ldlt.matrixL()*
      (ldlt.vectorD().cwiseMax(0.0).cwiseSqrt().asDiagonal()*B1));
}

template<class T>
void cholouter(const T& x,
    const T* S, const int ldS, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto S1 = make_eigen_matrix(S, n, n, ldS);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  C1.noalias() = (ldlt.transpositionsP().transpose()*(ldlt.matrixL()*
      (ldlt.vectorD().cwiseMax(0.0).cwiseSqrt().asDiagonal()*
      A1.transpose()))).transpose();
}

template<class T>
void cholsolve(const int n, const T* S, const int ldS, T* x, const int incx,
    const T* y, const int incy) {
  auto S1 = make_eigen_matrix(S, n, n, ldS);
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  x1.noalias() = ldlt.solve(y1);
}

template<class T>
void cholsolve(const T* S, const int ldS, T* X,
    const int ldX, const T* Y, const int ldY) {
  auto S1 = make_eigen_matrix(S, m, m, ldS);
  auto X1 = make_eigen_matrix(X, m, n, ldX);
  auto Y1 = make_eigen_matrix(Y, m, n, ldY);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  X1.noalias() = ldlt.solve(Y1);
}

template<class T>
void copysign(const T& x, const U& y, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, copysign_functor<T>());
}

template<class T>
void digamma(const T& x,
    const int* B, const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b).template cast<T>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, digammap_functor<T>());
}

template<class T>
void dot(const int n, const T* x, const int incx, const T* y, const int incy,
    T* z) {
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  *z = x1.dot(y1);
}

template<class T>
void frobenius(const T& x,
    const T* B, const int ldB, T* c) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  *c = (A1.array()*B1.array()).sum();
}

template<class T>
void gamma_p(const T& x, const U& y, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, gamma_p_functor<T>());
}

template<class T>
void gamma_q(const T& x, const U& y, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, gamma_q_functor<T>());
}

template<class T, class U, class>
void hadamard(const T& x, const U& y) {
  auto A1 = make_eigen_matrix(x).template cast<V>();
  auto B1 = make_eigen_matrix(b).template cast<V>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.cwiseProduct(B1);
}

template<class T>
void inner(const T& x, const T* x,
    const int incx, T* y, const int incy) {
  auto A1 = make_eigen_matrix(A, n, m, ldA);
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, m, incy);
  y1.noalias() = A1.transpose()*x1;
}

template<class T>
void inner(const int k, const T& x,
    const T* B, const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, k, m, ldA);
  auto B1 = make_eigen_matrix(B, k, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.transpose()*B1;
}

template<class T>
void lbeta(const T& x,
    const T* B, const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, lbeta_functor<T>());
}

template<class T>
void lchoose(const T& x,
    const int* B, const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(x).template cast<T>();
  auto B1 = make_eigen_matrix(b).template cast<T>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, lchoose_functor<T>());
}

template<class T>
void lchoose_grad(const T* G, const int ldG,
    const T& x, const int* B, const int ldB, T* GA,
    const int ldGA, T* GB, const int ldGB) {
  ///@todo Implement a generic ternary transform for this purpose
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      auto pair = lchoose_grad(G[i + j*ldG], A[i + j*ldA], B[i + j*ldB]);
      GA[i + j*ldGA] = pair.first;
      GB[i + j*ldGB] = pair.second;
    }
  }
}

template<class T>
void lgamma(const T& x, const U& y, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b).template cast<T>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, lgammap_functor<T>());
}

template<class T>
void outer(const T* x, const int incx, const T* y,
    const int incy, T* A, const int ldA) {
  auto x1 = make_eigen_vector(x, m, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  auto A1 = make_eigen_matrix(x);
  A1.noalias() = x1*y1.transpose();
}

template<class T>
void outer(const int k, const T& x,
    const T* B, const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, k, ldA);
  auto B1 = make_eigen_matrix(B, n, k, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1*B1.transpose();
}

template<class T>
void pow(const T& x, const U& y, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(x);
  auto B1 = make_eigen_matrix(b);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, pow_functor<T>());
}

template<class T>
void single(const int* i, const int* j, T* A,
    const int ldA) {
  auto A1 = make_eigen_matrix(x);
  A1.noalias() = A1.Zero(m, n);
  A1(*i - 1, *j - 1) = T(1);
}

template<class T>
void solve(const int n, const T& x, T* x, const int incx,
    const T* y, const int incy) {
  auto A1 = make_eigen_matrix(A, n, n, ldA);
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  x1.noalias() = A1.householderQr().solve(y1);
}

template<class T>
void solve(const T& x, T* X,
    const int ldX, const T* Y, const int ldY) {
  auto A1 = make_eigen_matrix(A, m, m, ldA);
  auto X1 = make_eigen_matrix(X, m, n, ldX);
  auto Y1 = make_eigen_matrix(Y, m, n, ldY);
  X1.noalias() = A1.householderQr().solve(Y1);
}

template<class T, class U>
void ibeta(const T& x, const U& y, const T* X, const int ldX, T* C, const int ldC) {
  ///@todo Implement a generic ternary transform for this purpose
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i + j*ldC] = ibeta(A[i + j*ldA], B[i + j*ldB], X[i + j*ldX]);
    }
  }
}

}
