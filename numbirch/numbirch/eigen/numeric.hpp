/**
 * @file
 */
#pragma once

#include "numbirch/eigen/eigen.hpp"

namespace numbirch {

void logical_not(const int m, const int n, const bool* A, const int ldA,
    bool* B, const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = A1.unaryExpr(logical_not_functor());
}

template<class T>
void neg(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = -A1;
}

template<class T, class U, class V>
void add(const int m, const int n, const T* A, const int ldA, const U* B,
    const int ldB, V* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA).template cast<V>();
  auto B1 = make_eigen_matrix(B, m, n, ldB).template cast<V>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1 + B1;
}

template<class T, class U, class V>
void div(const int m, const int n, const T* A, const int ldA, const U* b,
    V* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA).template cast<V>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1/(*b);
}

template<class T>
void equal(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, equal_functor<T>());
}

template<class T>
void greater(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, greater_functor<T>());
}

template<class T>
void greater_or_equal(const int m, const int n, const T* A, const int ldA,
    const T* B, const int ldB, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, greater_or_equal_functor<T>());
}

template<class T>
void less(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, less_functor<T>());
}

template<class T>
void less_or_equal(const int m, const int n, const T* A, const int ldA,
    const T* B, const int ldB, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, less_or_equal_functor<T>());
}

void logical_and(const int m, const int n, const bool* A, const int ldA,
    const bool* B, const int ldB, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, logical_and_functor());
}

void logical_or(const int m, const int n, const bool* A, const int ldA,
    const bool* B, const int ldB, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, logical_or_functor());
}

template<class T, class U, class V>
void mul(const int m, const int n, const T* a, const U* B, const int ldB,
    V* C, const int ldC) {
  auto B1 = make_eigen_matrix(B, m, n, ldB).template cast<V>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = (*a)*B1;
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
void not_equal(const int m, const int n, const T* A, const int ldA,
    const T* B, const int ldB, bool* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.binaryExpr(B1, not_equal_functor<T>());
}

template<class T, class U, class V>
void sub(const int m, const int n, const T* A, const int ldA, const U* B,
    const int ldB, V* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA).template cast<V>();
  auto B1 = make_eigen_matrix(B, m, n, ldB).template cast<V>();
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1 - B1;
}

template<class T>
void abs(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().abs();
}

template<class T>
void acos(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().acos();
}

template<class T>
void asin(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().asin();
}

template<class T>
void atan(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().atan();
}

template<class T>
void ceil(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
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
void cos(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().cos();
}

template<class T>
void cosh(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().cosh();
}

template<class T>
void count(const int m, const int n, const T* A, const int ldA, int* b) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  *b = A1.unaryExpr(count_functor<T>()).sum();
}

template<class T>
void diagonal(const T* a, const int n, T* B, const int ldB) {
  auto B1 = make_eigen_matrix(B, n, n, ldB);
  B1.noalias() = (*a)*B1.Identity(n, n);
}

template<class T>
void digamma(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = A1.unaryExpr(digamma_functor<T>());
}

template<class T>
void exp(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().exp();
}

template<class T>
void expm1(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = A1.unaryExpr(expm1_functor<T>());
}

template<class T>
void floor(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().floor();
}

template<class T>
void inv(const int n, const T* A, const int ldA, T* B, const int ldB) {
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
void ldet(const int n, const T* A, const int ldA, T* b) {
  auto A1 = make_eigen_matrix(A, n, n, ldA);
  *b = A1.householderQr().logAbsDeterminant();
}

template<class T>
void lfact(const int m, const int n, const int* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = A1.unaryExpr(lfact_functor<T>());
}

template<class T>
void lfact_grad(const int m, const int n, const T* G, const int ldG,
    const int* A, const int ldA, T* B, const int ldB) {
  auto G1 = make_eigen_matrix(G, m, n, ldG);
  auto A1 = make_eigen_matrix(A, m, n, ldA).template cast<T>();
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = G1.binaryExpr(A1, lfact_grad_functor<T>());
}

template<class T>
void lgamma(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = A1.unaryExpr(lgamma_functor<T>());
}

template<class T>
void log(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().log();
}

template<class T>
void log1p(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().log1p();
}

template<class T>
void rcp(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = A1.unaryExpr(rcp_functor<T>());
}

template<class T>
void rectify(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().max(T(0));
}

template<class T>
void rectify_grad(const int m, const int n, const T* G, const int ldG,
    const T* A, const int ldA, T* B, const int ldB) {
  auto G1 = make_eigen_matrix(G, m, n, ldG);
  auto A1 = make_eigen_matrix(A, m, n, ldA).template cast<T>();
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = G1.binaryExpr(A1, rectify_grad_functor<T>());
}

template<class T>
void round(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().round();
}

template<class T>
void sin(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().sin();
}

template<class T>
void single(const int* i, const int n, T* x, const int incx) {
  auto x1 = make_eigen_vector(x, n, incx);
  x1.noalias() = x1.Zero(n, 1);
  x1(*i - 1) = T(1);
}

template<class T>
void sinh(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().sinh();
}

template<class T>
void sqrt(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().sqrt();
}

template<class T>
void sum(const int m, const int n, const T* A, const int ldA, T* b) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  *b = A1.sum();
}

template<class T>
void tan(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().tan();
}

template<class T>
void tanh(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.array() = A1.array().tanh();
}

template<class T>
void trace(const int m, const int n, const T* A, const int ldA, T* b) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  *b = A1.trace();
}

template<class T>
void transpose(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, n, m, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
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
  A1(*i - 1, *j - 1) = T(1);
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

template<class T, class U>
void ibeta(const int m, const int n, const U* A, const int ldA, const U* B,
    const int ldB, const T* X, const int ldX, T* C, const int ldC) {
  ///@todo Implement a generic ternary transform for this purpose
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i + j*ldC] = ibeta(A[i + j*ldA], B[i + j*ldB], X[i + j*ldX]);
    }
  }
}

template<class T>
void combine(const int m, const int n, const T a, const T* A, const int ldA,
    const T b, const T* B, const int ldB, const T c, const T* C,
    const int ldC, const T d, const T* D, const int ldD, T* E,
    const int ldE) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  auto D1 = make_eigen_matrix(D, m, n, ldD);
  auto E1 = make_eigen_matrix(E, m, n, ldE);
  E1.noalias() = a*A1 + b*B1 + c*C1 + d*D1;
}

}
