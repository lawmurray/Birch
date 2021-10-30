/**
 * @file
 */
#pragma once

#include "numbirch/numeric/unary_function.hpp"
#include "numbirch/functor/unary_function.hpp"
#include "numbirch/eigen/eigen.hpp"

namespace numbirch {

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

}
