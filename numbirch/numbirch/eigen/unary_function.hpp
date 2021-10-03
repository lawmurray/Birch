/**
 * @file
 */
#pragma once

#include "numbirch/numeric/unary_function.hpp"
#include "numbirch/eigen/eigen.hpp"

namespace numbirch {

template<class T>
void cholinv(const int n, const T* S, const int ldS, T* B, const int ldB) {
  auto S1 = make_eigen_matrix(S, n, n, ldS);
  auto B1 = make_eigen_matrix(B, n, n, ldB);
  auto ldlt = S1.ldlt();
  assert(ldlt.info() == Eigen::Success);
  B1.noalias() = ldlt.solve(B1.Identity(n, n));
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
void rectify(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = A1.cwiseMax(T(0));
}

template<class T>
void sum(const int m, const int n, const T* A, const int ldA, T* b) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  *b = A1.sum();
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
