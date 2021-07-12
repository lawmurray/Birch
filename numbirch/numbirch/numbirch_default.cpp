/**
 * @file
 */
#include "numbirch/numbirch.hpp"

#include <eigen3/Eigen/Dense>

template<class T>
auto make_eigen_vector(T* x, const int n, const int incx) {
  using EigenVectorStride = Eigen::Stride<1,Eigen::Dynamic>;
  using EigenVector = Eigen::Matrix<typename std::remove_const<T>::type,
      Eigen::Dynamic,1,Eigen::ColMajor,Eigen::Dynamic,1>;
  using EigenVectorMap = Eigen::Map<typename std::conditional<
      std::is_const<T>::value,const EigenVector,EigenVector>::type,
      Eigen::DontAlign,EigenVectorStride>;
  return EigenVectorMap(x, n, 1, EigenVectorStride(1, incx));
}

template<class T>
auto make_eigen_matrix(T* A, const int m, const int n, const int ldA) {
  using EigenMatrixStride = Eigen::Stride<Eigen::Dynamic,1>;
  using EigenMatrix = Eigen::Matrix<typename std::remove_const<T>::type,
      Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor,Eigen::Dynamic,
      Eigen::Dynamic>;
  using EigenMatrixMap = Eigen::Map<typename std::conditional<
      std::is_const<T>::value,const EigenMatrix,EigenMatrix>::type,
      Eigen::DontAlign,EigenMatrixStride>;
  return EigenMatrixMap(A, m, n, EigenMatrixStride(ldA, 1));
}

void numbirch::init() {
  /* older compiler versions that do not support thread-safe static local
   * variable initialization require Eigen to initialize such variables before
   * entering a parallel region */
  Eigen::initParallel();
}

void* numbirch::malloc(const size_t size) {
  return std::malloc(size);
}

void* numbirch::realloc(void* ptr, size_t oldsize, size_t newsize) {
  return std::realloc(ptr, newsize);
}

void numbirch::free(void* ptr) {
  std::free(ptr);
}

void numbirch::neg(const int n, const double* x, const int incx, double* y,
    const int incy) {
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  y1.noalias() = -x1;
}

void numbirch::neg(const int m, const int n, const double* A, const int ldA,
    double* B, const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = -A1;
}

void numbirch::add(const int n, const double* x, const int incx,
    const double* y, const int incy, double* z, const int incz) {
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  auto z1 = make_eigen_vector(z, n, incz);
  z1.noalias() = x1 + y1;
}

void numbirch::add(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1 + B1;
}

void numbirch::sub(const int n, const double* x, const int incx,
    const double* y, const int incy, double* z, const int incz) {
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  auto z1 = make_eigen_vector(z, n, incz);
  z1.noalias() = x1 - y1;
}

void numbirch::sub(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1 - B1;
}

void numbirch::hadamard(const int n, const double* x, const int incx,
    const double* y, const int incy, double* z, const int incz) {
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  auto z1 = make_eigen_vector(z, n, incz);
  z1.noalias() = x1.cwiseProduct(y1);
}

void numbirch::hadamard(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.cwiseProduct(B1);
}

void numbirch::div(const int n, const double* x, const int incx,
    const double y, double* z, const int incz) {
  auto x1 = make_eigen_vector(x, n, incx);
  auto z1 = make_eigen_vector(z, n, incz);
  z1.noalias() = x1/y;
}

void numbirch::div(const int m, const int n, const double* A, const int ldA,
    const double b, double* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1/b;
}

void numbirch::mul(const int n, const double x, const double* y,
    const int incy, double* z, const int incz) {
  auto y1 = make_eigen_vector(y, n, incy);
  auto z1 = make_eigen_vector(z, n, incz);
  z1.noalias() = x*y1;
}

void numbirch::mul(const int m, const int n, const double a, const double* B,
    const int ldB, double* C, const int ldC) {
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = a*B1;
}

void numbirch::mul(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, m, incy);
  y1.noalias() = A1*x1;
}

void numbirch::mul(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, k, ldA);
  auto B1 = make_eigen_matrix(B, k, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1*B1;
}

double numbirch::sum(const int n, const double* x, const int incx) {
  auto x1 = make_eigen_vector(x, n, incx);
  return x1.sum();
}

double numbirch::sum(const int m, const int n, const double* A,
    const int ldA) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  return A1.sum();
}

double numbirch::dot(const int n, const double* x, const int incx,
    const double* y, const int incy) {
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  return x1.dot(y1);
}

double numbirch::frobenius(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  return (A1.array()*B1.array()).sum();
}

void numbirch::inner(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy) {
  auto A1 = make_eigen_matrix(A, n, m, ldA);
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, m, incy);
  y1.noalias() = A1.transpose()*x1;
}

void numbirch::inner(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, k, m, ldA);
  auto B1 = make_eigen_matrix(B, k, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1.transpose()*B1;
}

void numbirch::outer(const int m, const int n, const double* x,
    const int incx, const double* y, const int incy, double* A,
    const int ldA) {
  auto x1 = make_eigen_vector(x, m, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  A1.noalias() = x1*y1.transpose();
}

void numbirch::outer(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_eigen_matrix(A, m, k, ldA);
  auto B1 = make_eigen_matrix(B, n, k, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  C1.noalias() = A1*B1.transpose();
}

void numbirch::solve(const int n, const double* A, const int ldA, double* x,
    const int incx, const double* y, const int incy) {
  auto A1 = make_eigen_matrix(A, n, n, ldA);
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  x1.noalias() = A1.householderQr().solve(y1);
}

void numbirch::solve(const int m, const int n, const double* A, const int ldA,
    double* X, const int ldX, const double* Y, const int ldY) {
  auto A1 = make_eigen_matrix(A, m, m, ldA);
  auto X1 = make_eigen_matrix(X, m, n, ldX);
  auto Y1 = make_eigen_matrix(Y, m, n, ldY);
  X1.noalias() = A1.householderQr().solve(Y1);
}

void numbirch::cholsolve(const int n, const double* S, const int ldS,
    double* x, const int incx, const double* y, const int incy) {
  auto S1 = make_eigen_matrix(S, n, n, ldS);
  auto x1 = make_eigen_vector(x, n, incx);
  auto y1 = make_eigen_vector(y, n, incy);
  auto llt = S1.llt();
  assert(llt.info() == Eigen::Success);
  x1.noalias() = llt.solve(y1);
}

void numbirch::cholsolve(const int m, const int n, const double* S,
    const int ldS, double* X, const int ldX, const double* Y, const int ldY) {
  auto S1 = make_eigen_matrix(S, m, m, ldS);
  auto X1 = make_eigen_matrix(X, m, n, ldX);
  auto Y1 = make_eigen_matrix(Y, m, n, ldY);
  auto llt = S1.llt();
  assert(llt.info() == Eigen::Success);
  X1.noalias() = llt.solve(Y1);
}

void numbirch::inv(const int n, const double* A, const int ldA, double* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, n, n, ldA);
  auto B1 = make_eigen_matrix(B, n, n, ldB);
  B1.noalias() = A1.inverse();
}

void numbirch::cholinv(const int n, const double* S, const int ldS, double* B,
    const int ldB) {
  auto S1 = make_eigen_matrix(S, n, n, ldS);
  auto B1 = make_eigen_matrix(B, n, n, ldB);
  auto llt = S1.llt();
  assert(llt.info() == Eigen::Success);
  B1.noalias() = llt.solve(Eigen::Matrix<double,Eigen::Dynamic,
      Eigen::Dynamic,Eigen::ColMajor>::Identity(n, n));
}

double numbirch::ldet(const int n, const double* A, const int ldA) {
  auto A1 = make_eigen_matrix(A, n, n, ldA);
  return A1.householderQr().logAbsDeterminant();
}

double numbirch::lcholdet(const int n, const double* S, const int ldS) {
  auto S1 = make_eigen_matrix(S, n, n, ldS);
  auto llt = S1.llt();
  assert(llt.info() == Eigen::Success);
  return 2.0*llt.matrixLLT().diagonal().array().log().sum();
}

void numbirch::chol(const int n, const double* S, const int ldS, double* L,
    const int ldL) {
  auto S1 = make_eigen_matrix(S, n, n, ldS);
  auto L1 = make_eigen_matrix(L, n, n, ldL);
  auto llt = S1.llt();
  assert(llt.info() == Eigen::Success);
  L1 = llt.matrixL();
}

void numbirch::transpose(const int m, const int n, const double x,
    const double* A, const int ldA, double* B, const int ldB) {
  auto A1 = make_eigen_matrix(A, n, m, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = x*A1.transpose();
}

double numbirch::trace(const int m, const int n, const double* A,
    const int ldA) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  return A1.trace();
}
