/**
 * @file
 *
 * Eigen Matrix types for use with Birch Array.
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/basic.hpp"

namespace libbirch {

using EigenVectorStride = Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>;
template<class Type>
using EigenVector = Eigen::Matrix<Type,Eigen::Dynamic,1,Eigen::ColMajor,Eigen::Dynamic,1>;
template<class Type>
using EigenVectorMap = Eigen::Map<EigenVector<Type>,Eigen::DontAlign,EigenVectorStride>;

using EigenMatrixStride = Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>;
template<class Type>
using EigenMatrix = Eigen::Matrix<Type,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor,Eigen::Dynamic,Eigen::Dynamic>;
template<class Type>
using EigenMatrixMap = Eigen::Map<EigenMatrix<Type>,Eigen::DontAlign,EigenMatrixStride>;

/*
 * Eigen type for an array type.
 */
template<class ArrayType>
struct eigen_type {
  using type = typename std::conditional<ArrayType::frame_type::count() == 2,
      EigenMatrixMap<typename ArrayType::value_type>,
    typename std::conditional<ArrayType::frame_type::count() == 1,
      EigenVectorMap<typename ArrayType::value_type>,
    void>::type>::type;
};

template<class ArrayType>
struct eigen_stride_type {
  using type = typename std::conditional<ArrayType::frame_type::count() == 2,
      EigenMatrixStride,
    typename std::conditional<ArrayType::frame_type::count() == 1,
      EigenVectorStride,
    void>::type>::type;
};

/*
 * Eigen and array type compatibility checks.
 */
template<class ArrayType, class EigenType>
struct is_eigen_compatible {
  static const bool value =
      std::is_same<typename ArrayType::value_type,typename EigenType::value_type>::value &&
          ((ArrayType::frame_type::count() == 1 && EigenType::ColsAtCompileTime == 1) ||
           (ArrayType::frame_type::count() == 2 && EigenType::ColsAtCompileTime == Eigen::Dynamic));
};

template<class ArrayType, class EigenType>
struct is_diagonal_compatible {
  static const bool value =
      std::is_same<typename ArrayType::value_type,typename EigenType::value_type>::value &&
          ArrayType::frame_type::count() == 2 && EigenType::ColsAtCompileTime == 1;
};

template<class ArrayType, class EigenType>
struct is_triangle_compatible {
  static const bool value =
      std::is_same<typename ArrayType::value_type,typename EigenType::value_type>::value &&
          ArrayType::frame_type::count() == 2 && EigenType::ColsAtCompileTime == Eigen::Dynamic;
};
}

namespace bi {
  namespace type {
using LLT = Eigen::LLT<libbirch::EigenMatrix<Real64>>;
  }
}
