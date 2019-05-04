/**
 * @file
 *
 * Wrappers for Eigen functions that preserve its lazy evaluation.
 */
#pragma once

#include "libbirch/Eigen.hpp"
#include "libbirch/Array.hpp"

namespace bi {

template<class EigenType>
auto norm(const EigenType& o) {
  return o.norm();
}

template<class Type, class Frame>
auto norm(const libbirch::Array<Type,Frame>& o) {
  return norm(o.toEigen());
}

template<class EigenType>
auto dot(const EigenType& o) {
  return o.squaredNorm();
}

template<class Type, class Frame>
auto dot(const libbirch::Array<Type,Frame>& o) {
  return dot(o.toEigen());
}

template<class EigenType1, class EigenType2>
typename EigenType1::value_type dot(const EigenType1& o1,
    const EigenType2& o2) {
  return o1.dot(o2);
}

template<class Type, class EigenType1, class Frame2>
Type dot(const EigenType1& o1, const libbirch::Array<Type,Frame2>& o2) {
  return dot(o1, o2.toEigen());
}

template<class Type, class Frame1, class EigenType2>
Type dot(const libbirch::Array<Type,Frame1>& o1, const EigenType2& o2) {
  return dot(o1.toEigen(), o2);
}

template<class Type, class Frame1, class Frame2>
Type dot(const libbirch::Array<Type,Frame1>& o1,
    const libbirch::Array<Type,Frame2>& o2) {
  return dot(o1.toEigen(), o2.toEigen());
}

template<class EigenType>
auto det(const EigenType& o) {
  return o.determinant();
}

template<class Type, class Frame>
auto det(const libbirch::Array<Type,Frame>& o) {
  return det(o.toEigen());
}

template<class EigenType>
auto trans(const EigenType& o) {
  return o.transpose();
}

template<class Type, class Frame>
auto trans(const libbirch::Array<Type,Frame>& o) {
  return trans(o.toEigen());
}

template<class EigenType>
auto inv(const EigenType& o) {
  return o.inverse();
}

template<class Type, class Frame>
auto inv(const libbirch::Array<Type,Frame>& o) {
  return inv(o.toEigen());
}

template<class EigenType1, class EigenType2>
auto solve(const EigenType1& o1, const EigenType2& o2) {
  return o1.householderQr().solve(o2);
}

template<class Type, class EigenType1, class Frame2>
auto solve(const EigenType1& o1, const libbirch::Array<Type,Frame2>& o2) {
  return solve(o1, o2.toEigen());
}

template<class Type, class Frame1, class EigenType2>
auto solve(const libbirch::Array<Type,Frame1>& o1, const EigenType2& o2) {
  return solve(o1.toEigen(), o2);
}

template<class Type, class Frame1, class Frame2>
auto solve(const libbirch::Array<Type,Frame1>& o1,
    const libbirch::Array<Type,Frame2>& o2) {
  return solve(o1.toEigen(), o2.toEigen());
}

}
