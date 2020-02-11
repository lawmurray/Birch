/**
 * @file
 *
 * Wrappers for Eigen functions that preserve its lazy evaluation.
 */
#pragma once

#include "libbirch/Eigen.hpp"
#include "libbirch/Array.hpp"
#include "libbirch/basic.hpp"

namespace bi {
/**
 * Dot product of vector with itself.
 */
template<class T>
auto dot(const Eigen::MatrixBase<T>& o) {
  return o.squaredNorm();
}

inline auto dot(const libbirch::DefaultArray<bi::type::Real64,1>& o) {
  return dot(o.toEigen());
}

/**
 * Dot product of vector with another.
 */
template<class T, class U>
auto dot(const Eigen::MatrixBase<T>& o1, const Eigen::MatrixBase<U>& o2) {
  return o1.dot(o2);
}

template<class T>
auto dot(const Eigen::MatrixBase<T>& o1, const libbirch::DefaultArray<bi::type::Real64,1>& o2) {
  return dot(o1, o2.toEigen());
}

template<class T>
auto dot(const libbirch::DefaultArray<bi::type::Real64,1>& o1, const Eigen::MatrixBase<T>& o2) {
  return dot(o1.toEigen(), o2);
}

inline auto dot(const libbirch::DefaultArray<bi::type::Real64,1>& o1, const libbirch::DefaultArray<bi::type::Real64,1>& o2) {
  return dot(o1.toEigen(), o2.toEigen());
}

/**
 * Norm of a vector.
 */
template<class T>
auto norm(const Eigen::MatrixBase<T>& o) {
  return o.norm();
}

inline auto norm(const libbirch::DefaultArray<bi::type::Real64,1>& o) {
  return norm(o.toEigen());
}

/**
 * Element-wise square root of a vector.
 */
template<class T>
auto sqrt(const Eigen::MatrixBase<T>& o) {
  return o.array().sqrt().matrix();
}

inline auto sqrt(const libbirch::DefaultArray<bi::type::Real64,1>& o) {
  return sqrt(o.toEigen());
}

/**
 * Transpose of a matrix (or vector).
 */
template<class T>
auto transpose(const Eigen::MatrixBase<T>& o) {
  return o.transpose();
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

inline auto transpose(const libbirch::DefaultArray<bi::type::Real64,2>& o) {
  return transpose(o.toEigen());
}

inline auto transpose(const libbirch::DefaultArray<bi::type::Real64,1>& o) {
  return transpose(o.toEigen());
}

/**
 * Convert a vector into a diagonal matrix.
 */
template<class T, std::enable_if_t<T::ColsAtCompileTime == 1, int> = 0>
auto diagonal(const Eigen::MatrixBase<T>& o) {
  return o.asDiagonal();
}

inline auto diagonal(const libbirch::DefaultArray<bi::type::Real64,1>& o) {
  return diagonal(o.toEigen());
}

/**
 * Extract the diagonal of a matrix as a vector.
 */
template<class T, std::enable_if_t<T::ColsAtCompileTime == Eigen::Dynamic, int> = 0>
auto diagonal(const Eigen::MatrixBase<T>& o) {
  return o.diagonal();
}

inline auto diagonal(const libbirch::DefaultArray<bi::type::Real64,2>& o) {
  return diagonal(o.toEigen());
}

/**
 * Trace of a matrix.
 */
template<class T>
auto trace(const Eigen::MatrixBase<T>& o) {
  return o.trace();
}

template<class T>
auto trace(const Eigen::LLT<T>& o) {
  return o.trace();
}

template<class T>
auto trace(const Eigen::DiagonalWrapper<T>& o) {
  return o.trace();
}

inline auto trace(const libbirch::DefaultArray<bi::type::Real64,2>& o) {
  return trace(o.toEigen());
}

/**
 * Determinant of a matrix.
 */
template<class T>
auto det(const Eigen::MatrixBase<T>& o) {
  return o.determinant();
}

template<class T>
auto det(const Eigen::LLT<T>& o) {
  return o.determinant();
}

template<class T>
auto det(const Eigen::DiagonalWrapper<T>& o) {
  return o.determinant();
}

inline auto det(const libbirch::DefaultArray<bi::type::Real64,2>& o) {
  return det(o.toEigen());
}

/**
 * Cholesky factor of a positive symmetric matrix.
 */
template<class T>
auto cholesky(const Eigen::LLT<T>& o) {
  return o.matrixL();
}

/**
 * Inverse of a matrix.
 */
template<class T>
auto inv(const Eigen::MatrixBase<T>& o) {
  return o.inverse();
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

inline auto inv(const libbirch::DefaultArray<bi::type::Real64,2>& o) {
  return inv(o.toEigen());
}

/**
 * Solution of equations.
 */
template<class T, class U>
auto solve(const Eigen::MatrixBase<T>& o1, const Eigen::MatrixBase<U>& o2) {
  return o1.householderQr().solve(o2).eval();
}

template<class T>
auto solve(const Eigen::MatrixBase<T>& o1, const libbirch::DefaultArray<bi::type::Real64,2>& o2) {
  return solve(o1, o2.toEigen());
}

template<class T>
auto solve(const Eigen::MatrixBase<T>& o1, const libbirch::DefaultArray<bi::type::Real64,1>& o2) {
  return solve(o1, o2.toEigen());
}

template<class T>
auto solve(const libbirch::DefaultArray<bi::type::Real64,2>& o1, const Eigen::MatrixBase<T>& o2) {
  return solve(o1.toEigen(), o2);
}

inline auto solve(const libbirch::DefaultArray<bi::type::Real64,2>& o1, const libbirch::DefaultArray<bi::type::Real64,2>& o2) {
  return solve(o1.toEigen(), o2.toEigen());
}

inline auto solve(const libbirch::DefaultArray<bi::type::Real64,2>& o1, const libbirch::DefaultArray<bi::type::Real64,1>& o2) {
  return solve(o1.toEigen(), o2.toEigen());
}

template<class T, class U>
auto solve(const Eigen::LLT<T>& o1, const Eigen::MatrixBase<U>& o2) {
  return o1.solve(o2).eval();
}

template<class T>
auto solve(const Eigen::LLT<T>& o1, const libbirch::DefaultArray<bi::type::Real64,2>& o2) {
  return solve(o1, o2.toEigen());
}

template<class T>
auto solve(const Eigen::LLT<T>& o1, const libbirch::DefaultArray<bi::type::Real64,1>& o2) {
  return solve(o1, o2.toEigen());
}

template<class T, class U>
auto solve(const Eigen::DiagonalWrapper<T>& o1, const Eigen::MatrixBase<U>& o2) {
  return o1.inverse()*o2;
}

template<class T>
auto solve(const Eigen::DiagonalWrapper<T>& o1, const libbirch::DefaultArray<bi::type::Real64,2>& o2) {
  return solve(o1, o2.toEigen());
}

template<class T>
auto solve(const Eigen::DiagonalWrapper<T>& o1, const libbirch::DefaultArray<bi::type::Real64,1>& o2) {
  return solve(o1, o2.toEigen());
}

template<class T, unsigned Mode1, class U>
auto solve(const Eigen::TriangularView<T,Mode1>& o1, const Eigen::MatrixBase<U>& o2) {
  return o1.solve(o2).eval();
}

template<class T, unsigned Mode1>
auto solve(const Eigen::TriangularView<T,Mode1>& o1, const libbirch::DefaultArray<bi::type::Real64,2>& o2) {
  return solve(o1, o2.toEigen());
}

template<class T, unsigned Mode1>
auto solve(const Eigen::TriangularView<T,Mode1>& o1, const libbirch::DefaultArray<bi::type::Real64,1>& o2) {
  return solve(o1, o2.toEigen());
}

/**
 * Kronecker product.
 */
template<class T, class U>
auto kronecker(const Eigen::MatrixBase<T>& o1, const Eigen::MatrixBase<U>& o2) {
  return Eigen::kroneckerProduct(o1, o2);
}

template<class T>
auto kronecker(const Eigen::MatrixBase<T>& o1, const libbirch::DefaultArray<bi::type::Real64,2>& o2) {
  return kronecker(o1, o2.toEigen());
}

template<class T>
auto kronecker(const Eigen::MatrixBase<T>& o1, const libbirch::DefaultArray<bi::type::Real64,1>& o2) {
  return kronecker(o1, o2.toEigen());
}

template<class T>
auto kronecker(const libbirch::DefaultArray<bi::type::Real64,2>& o1, const Eigen::MatrixBase<T>& o2) {
  return kronecker(o1.toEigen(), o2);
}

template<class T>
auto kronecker(const libbirch::DefaultArray<bi::type::Real64,1>& o1, const Eigen::MatrixBase<T>& o2) {
  return kronecker(o1.toEigen(), o2);
}

inline auto kronecker(const libbirch::DefaultArray<bi::type::Real64,2>& o1, const libbirch::DefaultArray<bi::type::Real64,2>& o2) {
  return kronecker(o1.toEigen(), o2.toEigen());
}

inline auto kronecker(const libbirch::DefaultArray<bi::type::Real64,2>& o1, const libbirch::DefaultArray<bi::type::Real64,1>& o2) {
  return kronecker(o1.toEigen(), o2.toEigen());
}

inline auto kronecker(const libbirch::DefaultArray<bi::type::Real64,1>& o1, const libbirch::DefaultArray<bi::type::Real64,2>& o2) {
  return kronecker(o1.toEigen(), o2.toEigen());
}

inline auto kronecker(const libbirch::DefaultArray<bi::type::Real64,1>& o1, const libbirch::DefaultArray<bi::type::Real64,1>& o2) {
  return kronecker(o1.toEigen(), o2.toEigen());
}

/**
 * Hadamard product.
 */
template<class T, class U>
auto hadamard(const Eigen::MatrixBase<T>& o1, const Eigen::MatrixBase<U>& o2) {
  return o1.cwiseProduct(o2);
}

template<class T>
auto hadamard(const Eigen::MatrixBase<T>& o1, const libbirch::DefaultArray<bi::type::Real64,2>& o2) {
  return hadamard(o1, o2.toEigen());
}

template<class T>
auto hadamard(const libbirch::DefaultArray<bi::type::Real64,2>& o1, const Eigen::MatrixBase<T>& o2) {
  return hadamard(o1.toEigen(), o2);
}

inline auto hadamard(const libbirch::DefaultArray<bi::type::Real64,2>& o1, const libbirch::DefaultArray<bi::type::Real64,2>& o2) {
  return hadamard(o1.toEigen(), o2.toEigen());
}

template<class T>
auto hadamard(const libbirch::DefaultArray<bi::type::Real64,1>& o1, const Eigen::MatrixBase<T>& o2) {
  return hadamard(o1.toEigen(), o2);
}

inline auto hadamard(const libbirch::DefaultArray<bi::type::Real64,1>& o1, const libbirch::DefaultArray<bi::type::Real64,1>& o2) {
  return hadamard(o1.toEigen(), o2.toEigen());
}

}
