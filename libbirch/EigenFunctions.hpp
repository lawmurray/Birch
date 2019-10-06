/**
 * @file
 *
 * Wrappers for Eigen functions that preserve its lazy evaluation.
 */
#pragma once

#include "libbirch/Eigen.hpp"
#include "libbirch/Array.hpp"

namespace bi {

template<class Type>
auto norm(const Eigen::MatrixBase<Type>& o) {
  return o.norm();
}

template<class Type, class Frame>
auto norm(const libbirch::Array<Type,Frame>& o) {
  return norm(o.toEigen());
}

template<class Type>
auto dot(const Eigen::MatrixBase<Type>& o) {
  return o.squaredNorm();
}

template<class Type, class Frame>
auto dot(const libbirch::Array<Type,Frame>& o) {
  return dot(o.toEigen());
}

template<class Type1, class Type2>
auto dot(const Eigen::MatrixBase<Type1>& o1,
    const Eigen::MatrixBase<Type2>& o2) {
  return o1.dot(o2);
}

template<class Type1, class Type2, class Frame>
auto dot(const Eigen::MatrixBase<Type1>& o1,
    const libbirch::Array<Type2,Frame>& o2) {
  return dot(o1, o2.toEigen());
}

template<class Type1, class Frame, class Type2>
auto dot(const libbirch::Array<Type1,Frame>& o1,
    const Eigen::MatrixBase<Type2>& o2) {
  return dot(o1.toEigen(), o2);
}

template<class Type1, class Frame1, class Type2, class Frame2>
auto dot(const libbirch::Array<Type1,Frame1>& o1,
    const libbirch::Array<Type2,Frame2>& o2) {
  return dot(o1.toEigen(), o2.toEigen());
}

template<class Type>
auto trace(const Eigen::MatrixBase<Type>& o) {
  return o.trace();
}

template<class Type, class Frame>
auto trace(const libbirch::Array<Type,Frame>& o) {
  return trace(o.toEigen());
}

template<class Type>
auto trace(const Eigen::LLT<Type>& o) {
  return o.trace();
}

template<class Type>
auto trace(const Eigen::DiagonalWrapper<Type>& o) {
  return o.trace();
}

template<class Type>
auto det(const Eigen::MatrixBase<Type>& o) {
  return o.determinant();
}

template<class Type, class Frame>
auto det(const libbirch::Array<Type,Frame>& o) {
  return det(o.toEigen());
}

template<class Type>
auto det(const Eigen::LLT<Type>& o) {
  return o.determinant();
}

template<class Type>
auto det(const Eigen::DiagonalWrapper<Type>& o) {
  return o.determinant();
}

/**
 * Convert a vector into a diagonal matrix.
 */
template<class Type, std::enable_if_t<Type::ColsAtCompileTime == 1, int> = 0>
auto diagonal(const Eigen::MatrixBase<Type>& o) {
  return o.asDiagonal();
}

/**
 * Extract the diagonal of a matrix as a vector.
 */
template<class Type, std::enable_if_t<Type::ColsAtCompileTime == Eigen::Dynamic, int> = 0>
auto diagonal(const Eigen::MatrixBase<Type>& o) {
  return o.diagonal();
}

template<class Type, class Frame>
auto diagonal(const libbirch::Array<Type,Frame>& o) {
  return diagonal(o.toEigen());
}

template<class Type>
auto transpose(const Eigen::MatrixBase<Type>& o) {
  return o.transpose();
}

template<class Type, class Frame>
auto transpose(const libbirch::Array<Type,Frame>& o) {
  return transpose(o.toEigen());
}

template<class Type>
auto transpose(const Eigen::DiagonalWrapper<Type>& o) {
  return o.transpose();
}

template<class Type>
auto transpose(const Eigen::LLT<Type>& o) {
  return o.transpose();
}

template<class Type, unsigned Mode>
auto transpose(const Eigen::TriangularView<Type,Mode>& o) {
  return o.transpose();
}

template<class Type>
auto inv(const Eigen::MatrixBase<Type>& o) {
  return o.inverse();
}

template<class Type, class Frame>
auto inv(const libbirch::Array<Type,Frame>& o) {
  return inv(o.toEigen());
}

template<class Type>
auto inv(const Eigen::LLT<Type>& o) {
  return o.solve(libbirch::EigenMatrix<typename Type::value_type>::Identity(
      o.rows(), o.cols())).eval();
}

template<class Type>
auto inv(const Eigen::DiagonalWrapper<Type>& o) {
  return o.inverse();
}

template<class Type, unsigned Mode>
auto inv(const Eigen::TriangularView<Type,Mode>& o) {
  return o.inverse();
}

template<class Type1, class Type2>
auto solve(const Eigen::MatrixBase<Type1>& o1,
    const Eigen::MatrixBase<Type2>& o2) {
  return o1.householderQr().solve(o2).eval();
}

template<class Type1, class Type2, class Frame2>
auto solve(const Eigen::MatrixBase<Type1>& o1,
    const libbirch::Array<Type2,Frame2>& o2) {
  return solve(o1, o2.toEigen());
}

template<class Type1, class Frame1, class Type2>
auto solve(const libbirch::Array<Type1,Frame1>& o1,
    const Eigen::MatrixBase<Type2>& o2) {
  return solve(o1.toEigen(), o2);
}

template<class Type1, class Frame1, class Type2, class Frame2>
auto solve(const libbirch::Array<Type1,Frame1>& o1,
    const libbirch::Array<Type2,Frame2>& o2) {
  return solve(o1.toEigen(), o2.toEigen());
}

template<class Type1, class Type2>
auto solve(const Eigen::LLT<Type1>& o1, const Eigen::MatrixBase<Type2>& o2) {
  return o1.solve(o2).eval();
}

template<class Type1, class Type2, class Frame2>
auto solve(const Eigen::LLT<Type1>& o1,
    const libbirch::Array<Type2,Frame2>& o2) {
  return solve(o1, o2.toEigen());
}

template<class Type1, unsigned Mode1, class Type2>
auto solve(const Eigen::TriangularView<Type1,Mode1>& o1,
    const Eigen::MatrixBase<Type2>& o2) {
  return o1.solve(o2).eval();
}

template<class Type1, unsigned Mode1, class Type2, class Frame2>
auto solve(const Eigen::TriangularView<Type1,Mode1>& o1,
    const libbirch::Array<Type2,Frame2>& o2) {
  return solve(o1, o2.toEigen());
}

template<class Type>
auto cholesky(const Eigen::LLT<Type>& o) {
  return o.matrixL();
}

template<class Type1, class Type2>
auto kronecker(const Eigen::MatrixBase<Type1>& o1,
    const Eigen::MatrixBase<Type2>& o2) {
  return Eigen::kroneckerProduct(o1, o2);
}

template<class Type1, class Type2, class Frame2>
auto kronecker(const Eigen::MatrixBase<Type1>& o1,
    const libbirch::Array<Type2,Frame2>& o2) {
  return kronecker(o1, o2.toEigen());
}

template<class Type1, class Frame1, class Type2>
auto kronecker(const libbirch::Array<Type1,Frame1>& o1,
    const Eigen::MatrixBase<Type2>& o2) {
  return kronecker(o1.toEigen(), o2);
}

template<class Type1, class Frame1, class Type2, class Frame2>
auto kronecker(const libbirch::Array<Type1,Frame1>& o1,
    const libbirch::Array<Type2,Frame2>& o2) {
  return kronecker(o1.toEigen(), o2.toEigen());
}

template<class Type1, class Type2>
auto hadamard(const Eigen::MatrixBase<Type1>& o1,
    const Eigen::MatrixBase<Type2>& o2) {
  return o1.cwiseProduct(o2);
}

template<class Type1, class Type2, class Frame2>
auto hadamard(const Eigen::MatrixBase<Type1>& o1,
    const libbirch::Array<Type2,Frame2>& o2) {
  return hadamard(o1, o2.toEigen());
}

template<class Type1, class Frame1, class Type2>
auto hadamard(const libbirch::Array<Type1,Frame1>& o1,
    const Eigen::MatrixBase<Type2>& o2) {
  return hadamard(o1.toEigen(), o2);
}

template<class Type1, class Frame1, class Type2, class Frame2>
auto hadamard(const libbirch::Array<Type1,Frame1>& o1,
    const libbirch::Array<Type2,Frame2>& o2) {
  return hadamard(o1.toEigen(), o2.toEigen());
}

template<class Type>
auto sqrt(const Eigen::MatrixBase<Type>& o) {
  return o.array().sqrt().matrix();
}

template<class Type, class Frame>
auto sqrt(const libbirch::Array<Type,Frame>& o) {
  return sqrt(o.toEigen());
}

}
