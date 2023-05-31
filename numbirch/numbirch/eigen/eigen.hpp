/**
 * @file
 * 
 * Eigen integration.
 */
#pragma once

#include "numbirch/array.hpp"

#if defined(HAVE_EIGEN_DENSE)
#include <Eigen/Dense>
#elif defined(HAVE_EIGEN3_EIGEN_DENSE)
#include <eigen3/Eigen/Dense>
#endif

namespace numbirch {

template<class T>
static auto make_eigen(const T* ptr) {
  using EigenScalarStride = Eigen::Stride<1,1>;
  using EigenScalar = Eigen::Matrix<T,1,1,Eigen::ColMajor,1,1>;
  using EigenScalarMap = const Eigen::Map<EigenScalar,Eigen::DontAlign,
      EigenScalarStride>;
  return EigenScalarMap(ptr, 1, 1, EigenScalarStride(1, 1));
}

template<class T>
static auto make_eigen(T* ptr) {
  using EigenScalarStride = Eigen::Stride<1,1>;
  using EigenScalar = Eigen::Matrix<T,1,1,Eigen::ColMajor,1,1>;
  using EigenScalarMap = Eigen::Map<EigenScalar,Eigen::DontAlign,
      EigenScalarStride>;
  return EigenScalarMap(ptr, 1, 1, EigenScalarStride(1, 1));
}

template<class T>
static auto make_eigen(const T* ptr, const int n, const int inc) {
  using EigenVectorStride = Eigen::Stride<1,Eigen::Dynamic>;
  using EigenVector = const Eigen::Matrix<T,Eigen::Dynamic,1,Eigen::ColMajor,
      Eigen::Dynamic,1>;
  using EigenVectorMap = Eigen::Map<EigenVector,Eigen::DontAlign,
      EigenVectorStride>;
  return EigenVectorMap(ptr, n, 1, EigenVectorStride(1, inc));
}

template<class T>
static auto make_eigen(T* ptr, const int n, const int inc) {
  using EigenVectorStride = Eigen::Stride<1,Eigen::Dynamic>;
  using EigenVector = Eigen::Matrix<T,Eigen::Dynamic,1,Eigen::ColMajor,
      Eigen::Dynamic,1>;
  using EigenVectorMap = Eigen::Map<EigenVector,Eigen::DontAlign,
      EigenVectorStride>;
  return EigenVectorMap(ptr, n, 1, EigenVectorStride(1, inc));
}

template<class T>
static auto make_eigen(const T* ptr, const int m, const int n, const int ld) {
  using EigenMatrixStride = Eigen::Stride<Eigen::Dynamic,1>;
  using EigenMatrix = const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,
      Eigen::ColMajor,Eigen::Dynamic,Eigen::Dynamic>;
  using EigenMatrixMap = Eigen::Map<EigenMatrix,Eigen::DontAlign,
      EigenMatrixStride>;
  return EigenMatrixMap(ptr, m, n, EigenMatrixStride(ld, 1));
}

template<class T>
static auto make_eigen(T* ptr, const int m, const int n, const int ld) {
  using EigenMatrixStride = Eigen::Stride<Eigen::Dynamic,1>;
  using EigenMatrix = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,
      Eigen::ColMajor,Eigen::Dynamic,Eigen::Dynamic>;
  using EigenMatrixMap = Eigen::Map<EigenMatrix,Eigen::DontAlign,
      EigenMatrixStride>;
  return EigenMatrixMap(ptr, m, n, EigenMatrixStride(ld, 1));
}

template<class T>
static auto make_eigen(Array<T,0>& x) {
  return make_eigen(buffer(x));
}

template<class T>
static auto make_eigen(const Array<T,0>& x) {
  return make_eigen(buffer(x));
}

template<class T>
static auto make_eigen(Array<T,1>& x) {
  return make_eigen(buffer(x), length(x), stride(x));
}

template<class T>
static auto make_eigen(const Array<T,1>& x) {
  return make_eigen(buffer(x), length(x), stride(x));
}

template<class T>
static auto make_eigen(Array<T,2>& x) {
  return make_eigen(buffer(x), rows(x), columns(x), stride(x));
}

template<class T>
static auto make_eigen(const Array<T,2>& x) {
  return make_eigen(buffer(x), rows(x), columns(x), stride(x));
}

}
