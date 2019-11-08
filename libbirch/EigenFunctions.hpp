/**
 * @file
 *
 * Wrappers for Eigen functions that preserve its lazy evaluation.
 */
#pragma once

#include "libbirch/Eigen.hpp"
#include "libbirch/Array.hpp"

namespace bi {

template<class T>
auto norm(const Eigen::MatrixBase<T>& o) {
  return o.norm();
}

template<class T, class F>
auto norm(const libbirch::Array<T,F>& o) {
  return norm(o.toEigen());
}

template<class T>
auto dot(const Eigen::MatrixBase<T>& o) {
  return o.squaredNorm();
}

template<class T, class F>
auto dot(const libbirch::Array<T,F>& o) {
  return dot(o.toEigen());
}

template<class T, class U>
auto dot(const Eigen::MatrixBase<T>& o1,
    const Eigen::MatrixBase<U>& o2) {
  return o1.dot(o2);
}

template<class T, class U, class F>
auto dot(const Eigen::MatrixBase<T>& o1,
    const libbirch::Array<U,F>& o2) {
  return dot(o1, o2.toEigen());
}

template<class T, class F, class U>
auto dot(const libbirch::Array<T,F>& o1,
    const Eigen::MatrixBase<U>& o2) {
  return dot(o1.toEigen(), o2);
}

template<class T, class F, class U, class G>
auto dot(const libbirch::Array<T,F>& o1,
    const libbirch::Array<U,G>& o2) {
  return dot(o1.toEigen(), o2.toEigen());
}

template<class T>
auto trace(const Eigen::MatrixBase<T>& o) {
  return o.trace();
}

template<class T, class F>
auto trace(const libbirch::Array<T,F>& o) {
  return trace(o.toEigen());
}

template<class T>
auto trace(const Eigen::LLT<T>& o) {
  return o.trace();
}

template<class T>
auto trace(const Eigen::DiagonalWrapper<T>& o) {
  return o.trace();
}

template<class T>
auto det(const Eigen::MatrixBase<T>& o) {
  return o.determinant();
}

template<class T, class F>
auto det(const libbirch::Array<T,F>& o) {
  return det(o.toEigen());
}

template<class T>
auto det(const Eigen::LLT<T>& o) {
  return o.determinant();
}

template<class T>
auto det(const Eigen::DiagonalWrapper<T>& o) {
  return o.determinant();
}

/**
 * Convert a vector into a diagonal matrix.
 */
template<class T, std::enable_if_t<T::ColsAtCompileTime == 1, int> = 0>
auto diagonal(const Eigen::MatrixBase<T>& o) {
  return o.asDiagonal();
}

/**
 * Extract the diagonal of a matrix as a vector.
 */
template<class T, std::enable_if_t<T::ColsAtCompileTime == Eigen::Dynamic, int> = 0>
auto diagonal(const Eigen::MatrixBase<T>& o) {
  return o.diagonal();
}

template<class T, class F>
auto diagonal(const libbirch::Array<T,F>& o) {
  return diagonal(o.toEigen());
}

template<class T>
auto transpose(const Eigen::MatrixBase<T>& o) {
  return o.transpose();
}

template<class T, class F>
auto transpose(const libbirch::Array<T,F>& o) {
  return transpose(o.toEigen());
}

template<class T>
auto transpose(const Eigen::LLT<T>& o) {
  return o.transpose();
}

template<class T>
auto transpose(const Eigen::DiagonalWrapper<T>& o) {
  return o.transpose();
}

template<class T, unsigned Mode>
auto transpose(const Eigen::TriangularView<T,Mode>& o) {
  return o.transpose();
}

template<class T>
auto inv(const Eigen::MatrixBase<T>& o) {
  return o.inverse();
}

template<class T, class F>
auto inv(const libbirch::Array<T,F>& o) {
  return inv(o.toEigen());
}

template<class T>
auto inv(const Eigen::LLT<T>& o) {
  return o.solve(libbirch::EigenMatrix<typename T::value_type>::Identity(
      o.rows(), o.cols())).eval();
}

template<class T>
auto inv(const Eigen::DiagonalWrapper<T>& o) {
  return o.inverse();
}

template<class T, unsigned Mode>
auto inv(const Eigen::TriangularView<T,Mode>& o) {
  return o.inverse();
}

template<class T, class U>
auto solve(const Eigen::MatrixBase<T>& o1,
    const Eigen::MatrixBase<U>& o2) {
  return o1.householderQr().solve(o2).eval();
}

template<class T, class U, class G>
auto solve(const Eigen::MatrixBase<T>& o1,
    const libbirch::Array<U,G>& o2) {
  return solve(o1, o2.toEigen());
}

template<class T, class F, class U>
auto solve(const libbirch::Array<T,F>& o1,
    const Eigen::MatrixBase<U>& o2) {
  return solve(o1.toEigen(), o2);
}

template<class T, class F, class U, class G>
auto solve(const libbirch::Array<T,F>& o1,
    const libbirch::Array<U,G>& o2) {
  return solve(o1.toEigen(), o2.toEigen());
}

template<class T, class U>
auto solve(const Eigen::LLT<T>& o1, const Eigen::MatrixBase<U>& o2) {
  return o1.solve(o2).eval();
}

template<class T, class U, class G>
auto solve(const Eigen::LLT<T>& o1,
    const libbirch::Array<U,G>& o2) {
  return solve(o1, o2.toEigen());
}

template<class T, class U>
auto solve(const Eigen::DiagonalWrapper<T>& o1,
    const Eigen::MatrixBase<U>& o2) {
  return o1.inverse()*o2;
}

template<class T, class U, class G>
auto solve(const Eigen::DiagonalWrapper<T>& o1,
    const libbirch::Array<U,G>& o2) {
  return solve(o1, o2.toEigen());
}

template<class T, unsigned Mode1, class U>
auto solve(const Eigen::TriangularView<T,Mode1>& o1,
    const Eigen::MatrixBase<U>& o2) {
  return o1.solve(o2).eval();
}

template<class T, unsigned Mode1, class U, class G>
auto solve(const Eigen::TriangularView<T,Mode1>& o1,
    const libbirch::Array<U,G>& o2) {
  return solve(o1, o2.toEigen());
}

template<class T>
auto cholesky(const Eigen::LLT<T>& o) {
  return o.matrixL();
}

template<class T, class U>
auto kronecker(const Eigen::MatrixBase<T>& o1,
    const Eigen::MatrixBase<U>& o2) {
  return Eigen::kroneckerProduct(o1, o2);
}

template<class T, class U, class G>
auto kronecker(const Eigen::MatrixBase<T>& o1,
    const libbirch::Array<U,G>& o2) {
  return kronecker(o1, o2.toEigen());
}

template<class T, class F, class U>
auto kronecker(const libbirch::Array<T,F>& o1,
    const Eigen::MatrixBase<U>& o2) {
  return kronecker(o1.toEigen(), o2);
}

template<class T, class F, class U, class G>
auto kronecker(const libbirch::Array<T,F>& o1,
    const libbirch::Array<U,G>& o2) {
  return kronecker(o1.toEigen(), o2.toEigen());
}

template<class T, class U>
auto hadamard(const Eigen::MatrixBase<T>& o1,
    const Eigen::MatrixBase<U>& o2) {
  return o1.cwiseProduct(o2);
}

template<class T, class U, class G>
auto hadamard(const Eigen::MatrixBase<T>& o1,
    const libbirch::Array<U,G>& o2) {
  return hadamard(o1, o2.toEigen());
}

template<class T, class F, class U>
auto hadamard(const libbirch::Array<T,F>& o1,
    const Eigen::MatrixBase<U>& o2) {
  return hadamard(o1.toEigen(), o2);
}

template<class T, class F, class U, class G>
auto hadamard(const libbirch::Array<T,F>& o1,
    const libbirch::Array<U,G>& o2) {
  return hadamard(o1.toEigen(), o2.toEigen());
}

template<class T>
auto sqrt(const Eigen::MatrixBase<T>& o) {
  return o.array().sqrt().matrix();
}

template<class T, class F>
auto sqrt(const libbirch::Array<T,F>& o) {
  return sqrt(o.toEigen());
}

}
