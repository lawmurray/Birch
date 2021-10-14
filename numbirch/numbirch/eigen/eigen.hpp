/**
 * @file
 * 
 * Eigen integration.
 */
#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/SpecialFunctions>

template<class T>
static auto make_eigen_vector(T* x, const int n, const int incx) {
  using EigenVectorStride = Eigen::Stride<1,Eigen::Dynamic>;
  using EigenVector = Eigen::Matrix<typename std::remove_const<T>::type,
      Eigen::Dynamic,1,Eigen::ColMajor,Eigen::Dynamic,1>;
  using EigenVectorMap = Eigen::Map<typename std::conditional<
      std::is_const<T>::value,const EigenVector,EigenVector>::type,
      Eigen::DontAlign,EigenVectorStride>;
  return EigenVectorMap(x, n, 1, EigenVectorStride(1, incx));
}

template<class T>
static auto make_eigen_matrix(T* A, const int m, const int n, const int ldA) {
  using EigenMatrixStride = Eigen::Stride<Eigen::Dynamic,1>;
  using EigenMatrix = Eigen::Matrix<typename std::remove_const<T>::type,
      Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor,Eigen::Dynamic,
      Eigen::Dynamic>;
  using EigenMatrixMap = Eigen::Map<typename std::conditional<
      std::is_const<T>::value,const EigenMatrix,EigenMatrix>::type,
      Eigen::DontAlign,EigenMatrixStride>;
  return EigenMatrixMap(A, m, n, EigenMatrixStride(ldA, 1));
}
