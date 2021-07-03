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
      Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor,Eigen::Dynamic,
      Eigen::Dynamic>;
  using EigenMatrixMap = Eigen::Map<typename std::conditional<
      std::is_const<T>::value,const EigenMatrix,EigenMatrix>::type,
      Eigen::DontAlign,EigenMatrixStride>;
  return EigenMatrixMap(A, m, n, EigenMatrixStride(ldA, 1));
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
