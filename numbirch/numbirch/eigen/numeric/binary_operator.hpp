/**
 * @file
 */
#pragma once

#include "numbirch/numeric/binary_operator.hpp"
#include "numbirch/functor/binary_operator.hpp"
#include "numbirch/eigen/eigen.hpp"

namespace numbirch {

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

}
