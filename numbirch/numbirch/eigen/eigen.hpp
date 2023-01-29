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
static auto make_eigen(Array<T,0>& x) {
  using EigenVectorStride = Eigen::Stride<1,1>;
  using EigenVector = Eigen::Matrix<T,1,1,Eigen::ColMajor,1,1>;
  using EigenVectorMap = Eigen::Map<EigenVector,Eigen::DontAlign,
      EigenVectorStride>;
  return EigenVectorMap(sliced(x), 1, 1, EigenVectorStride(1, 1));
}

template<class T>
static auto make_eigen(const Array<T,0>& x) {
  using EigenVectorStride = Eigen::Stride<1,1>;
  using EigenVector = const Eigen::Matrix<T,1,1,Eigen::ColMajor,1,1>;
  using EigenVectorMap = Eigen::Map<EigenVector,Eigen::DontAlign,
      EigenVectorStride>;
  return EigenVectorMap(sliced(x), 1, 1, EigenVectorStride(1, 1));
}

template<class T>
static auto make_eigen(Array<T,1>& x) {
  using EigenVectorStride = Eigen::Stride<1,Eigen::Dynamic>;
  using EigenVector = Eigen::Matrix<T,Eigen::Dynamic,
      1,Eigen::ColMajor,Eigen::Dynamic,1>;
  using EigenVectorMap = Eigen::Map<EigenVector,Eigen::DontAlign,
      EigenVectorStride>;
  return EigenVectorMap(sliced(x), length(x), 1,
      EigenVectorStride(1, stride(x)));
}

template<class T>
static auto make_eigen(const Array<T,1>& x) {
  using EigenVectorStride = Eigen::Stride<1,Eigen::Dynamic>;
  using EigenVector = const Eigen::Matrix<T,Eigen::Dynamic,
      1,Eigen::ColMajor,Eigen::Dynamic,1>;
  using EigenVectorMap = Eigen::Map<EigenVector,Eigen::DontAlign,
      EigenVectorStride>;
  return EigenVectorMap(sliced(x), length(x), 1,
      EigenVectorStride(1, stride(x)));
}

template<class T>
static auto make_eigen(Array<T,2>& x) {
  using EigenMatrixStride = Eigen::Stride<Eigen::Dynamic,1>;
  using EigenMatrix = Eigen::Matrix<T,Eigen::Dynamic,
      Eigen::Dynamic,Eigen::ColMajor,Eigen::Dynamic,Eigen::Dynamic>;
  using EigenMatrixMap = Eigen::Map<EigenMatrix,Eigen::DontAlign,
      EigenMatrixStride>;
  return EigenMatrixMap(sliced(x), rows(x), columns(x),
      EigenMatrixStride(stride(x), 1));
}

template<class T>
static auto make_eigen(const Array<T,2>& x) {
  using EigenMatrixStride = Eigen::Stride<Eigen::Dynamic,1>;
  using EigenMatrix = const Eigen::Matrix<T,Eigen::Dynamic,
      Eigen::Dynamic,Eigen::ColMajor,Eigen::Dynamic,Eigen::Dynamic>;
  using EigenMatrixMap = Eigen::Map<EigenMatrix,Eigen::DontAlign,
      EigenMatrixStride>;
  return EigenMatrixMap(sliced(x), rows(x), columns(x),
      EigenMatrixStride(stride(x), 1));
}

}
