/**
 * @file
 */
#pragma once

#include "numbirch/numeric/binary_function.hpp"
#include "numbirch/functor/binary_function.hpp"
#include "numbirch/eigen/eigen.hpp"

namespace numbirch {

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
void cholmul(const int m, const int n, const T* S, const int ldS, const T* B,
    const int ldB, T* C, const int ldC) {
  auto S1 = make_eigen_matrix(S, m, m, ldS);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  C1.noalias() = ldlt.transpositionsP().transpose()*(ldlt.matrixL()*
      (ldlt.vectorD().cwiseMax(0.0).cwiseSqrt().asDiagonal()*B1));
}

template<class T>
void cholouter(const int m, const int n, const T* A, const int ldA,
    const T* S, const int ldS, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
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
void cholsolve(const int m, const int n, const T* S, const int ldS, T* X,
    const int ldX, const T* Y, const int ldY) {
  auto S1 = make_eigen_matrix(S, m, m, ldS);
  auto X1 = make_eigen_matrix(X, m, n, ldX);
  auto Y1 = make_eigen_matrix(Y, m, n, ldY);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  X1.noalias() = ldlt.solve(Y1);
}

template<class T>
void copysign(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, copysign_functor<T>());
}

template<class T>
void digamma(const int m, const int n, const T* A, const int ldA,
    const int* B, const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB).template cast<T>();
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
void frobenius(const int m, const int n, const T* A, const int ldA,
    const T* B, const int ldB, T* c) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  *c = (A1.array()*B1.array()).sum();
}

template<class T>
void gamma_p(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, gamma_p_functor<T>());
}

template<class T>
void gamma_q(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, gamma_q_functor<T>());
}

template<class T, class U, class V>
void hadamard(const int m, const int n, const T* A, const int ldA, const U* B,
    const int ldB, V* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA).template cast<V>();
  auto B1 = make_eigen_matrix(B, m, n, ldB).template cast<V>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.cwiseProduct(B1);
}

template<class T>
void inner(const int m, const int n, const T* A, const int ldA, const T* x,
    const int incx, T* y, const int incy) {
  auto A1 = make_eigen_matrix(A, n, m, ldA);
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, m, incy);
  y1.noalias() = A1.transpose()*x1;
}

template<class T>
void inner(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, k, m, ldA);
  auto B1 = make_eigen_matrix(B, k, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.transpose()*B1;
}

template<class T>
void lbeta(const int m, const int n, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, lbeta_functor<T>());
}

template<class T>
void lchoose(const int m, const int n, const int* A, const int ldA,
    const int* B, const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA).template cast<T>();
  auto B1 = make_eigen_matrix(B, m, n, ldB).template cast<T>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, lchoose_functor<T>());
}

template<class T>
void lchoose_grad(const int m, const int n, const T* G, const int ldG,
    const int* A, const int ldA, const int* B, const int ldB, T* GA,
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
void lgamma(const int m, const int n, const T* A, const int ldA, const int* B,
    const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB).template cast<T>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, lgammap_functor<T>());
}

template<class T>
void outer(const int m, const int n, const T* x, const int incx, const T* y,
    const int incy, T* A, const int ldA) {
  auto x1 = make_eigen_vector(x, m, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  A1.noalias() = x1*y1.transpose();
}

template<class T>
void outer(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, k, ldA);
  auto B1 = make_eigen_matrix(B, n, k, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1*B1.transpose();
}

template<class T>
void pow(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, pow_functor<T>());
}

template<class T>
void single(const int* i, const int* j, const int m, const int n, T* A,
    const int ldA) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  A1.noalias() = A1.Zero(m, n);
  A1(*i, *j) = T(1);
}

template<class T>
void solve(const int n, const T* A, const int ldA, T* x, const int incx,
    const T* y, const int incy) {
  auto A1 = make_eigen_matrix(A, n, n, ldA);
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  x1.noalias() = A1.householderQr().solve(y1);
}

template<class T>
void solve(const int m, const int n, const T* A, const int ldA, T* X,
    const int ldX, const T* Y, const int ldY) {
  auto A1 = make_eigen_matrix(A, m, m, ldA);
  auto X1 = make_eigen_matrix(X, m, n, ldX);
  auto Y1 = make_eigen_matrix(Y, m, n, ldY);
  X1.noalias() = A1.householderQr().solve(Y1);
}

}
