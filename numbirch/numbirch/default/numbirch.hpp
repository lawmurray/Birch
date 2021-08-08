/**
 * @file
 * 
 * Generic implementation of interface for Eigen.
 */
#pragma once

#include "numbirch/default/eigen.hpp"

namespace numbirch {

template<class T>
void neg(const int n, const T* x, const int incx, T* y, const int incy) {
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  y1.noalias() = -x1;
}

template<class T>
void neg(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = -A1;
}

template<class T>
void add(const int n, const T* x, const int incx, const T* y, const int incy,
    T* z, const int incz) {
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  auto z1 = make_eigen_vector(z, n, incz);
  z1.noalias() = x1 + y1;
}

template<class T>
void add(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1 + B1;
}

template<class T>
void sub(const int n, const T* x, const int incx, const T* y, const int incy,
    T* z, const int incz) {
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  auto z1 = make_eigen_vector(z, n, incz);
  z1.noalias() = x1 - y1;
}

template<class T>
void sub(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1 - B1;
}

template<class T>
void hadamard(const int n, const T* x, const int incx, const T* y,
    const int incy, T* z, const int incz) {
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  auto z1 = make_eigen_vector(z, n, incz);
  z1.noalias() = x1.cwiseProduct(y1);
}

template<class T>
void hadamard(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.cwiseProduct(B1);
}

template<class T>
void div(const int n, const T* x, const int incx, const T y, T* z,
    const int incz) {
  auto x1 = make_eigen_vector(x, n, incx);
  auto z1 = make_eigen_vector(z, n, incz);
  z1.noalias() = x1/y;
}

template<class T>
void div(const int m, const int n, const T* A, const int ldA, const T b, T* C,
    const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1/b;
}

template<class T>
void mul(const int n, const T x, const T* y, const int incy, T* z,
    const int incz) {
  auto y1 = make_eigen_vector(y, n, incy);
  auto z1 = make_eigen_vector(z, n, incz);
  z1.noalias() = x*y1;
}

template<class T>
void mul(const int m, const int n, const T a, const T* B, const int ldB, T* C,
    const int ldC) {
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = a*B1;
}

template<class T>
void mul(const int m, const int n, const T* A, const int ldA, const T* x,
    const int incx, T* y, const int incy) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, m, incy);
  y1.noalias() = A1*x1;
}

template<class T>
void mul(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, k, ldA);
  auto B1 = make_eigen_matrix(B, k, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1*B1;
}

template<class T>
void cholmul(const int n, const T* S, const int ldS, const T* x,
    const int incx, T* y, const int incy) {
  auto S1 = make_eigen_matrix(S, n, n, ldS);
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  auto ldlt = S1.ldlt();
  //assert(ldlt.info() == Eigen::Success);
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
  //assert(ldlt.info() == Eigen::Success);
  C1.noalias() = ldlt.transpositionsP().transpose()*(ldlt.matrixL()*
      (ldlt.vectorD().cwiseMax(0.0).cwiseSqrt().asDiagonal()*B1));
}

template<class T>
T sum(const int n, const T* x, const int incx) {
  auto x1 = make_eigen_vector(x, n, incx);
  return x1.sum();
}

template<class T>
T sum(const int m, const int n, const T* A, const int ldA) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  return A1.sum();
}

template<class T>
T dot(const int n, const T* x, const int incx, const T* y, const int incy) {
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  return x1.dot(y1);
}

template<class T>
T frobenius(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  return (A1.array()*B1.array()).sum();
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
void cholouter(const int m, const int n, const T* A, const int ldA,
    const T* S, const int ldS, T* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto S1 = make_eigen_matrix(S, n, n, ldS);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  auto ldlt = S1.ldlt();
  //assert(ldlt.info() == Eigen::Success);
  C1.noalias() = (ldlt.transpositionsP().transpose()*(ldlt.matrixL()*
      (ldlt.vectorD().cwiseMax(0.0).cwiseSqrt().asDiagonal()*
      A1.transpose()))).transpose();
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
void inv(const int n, const T* A, const int ldA, T* B, const int ldB) {
  auto A1 = make_eigen_matrix(A, n, n, ldA);
  auto B1 = make_eigen_matrix(B, n, n, ldB);
  B1.noalias() = A1.inverse();
}

template<class T>
void cholinv(const int n, const T* S, const int ldS, T* B, const int ldB) {
  auto S1 = make_eigen_matrix(S, n, n, ldS);
  auto B1 = make_eigen_matrix(B, n, n, ldB);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  B1.noalias() = ldlt.solve(Eigen::Matrix<T,Eigen::Dynamic,
      Eigen::Dynamic,Eigen::ColMajor>::Identity(n, n));
}

template<class T>
T ldet(const int n, const T* A, const int ldA) {
  auto A1 = make_eigen_matrix(A, n, n, ldA);
  return A1.householderQr().logAbsDeterminant();
}

template<class T>
T lcholdet(const int n, const T* S, const int ldS) {
  auto S1 = make_eigen_matrix(S, n, n, ldS);
  auto llt = S1.llt();
  assert(llt.info() == Eigen::Success);
  return 2.0*llt.matrixLLT().diagonal().array().log().sum();
}

template<class T>
void transpose(const int m, const int n, const T x, const T* A, const int ldA,
    T* B, const int ldB) {
  auto A1 = make_eigen_matrix(A, n, m, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = x*A1.transpose();
}

template<class T>
T trace(const int m, const int n, const T* A, const int ldA) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  return A1.trace();
}

}
