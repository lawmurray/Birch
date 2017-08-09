/**
 * @file
 *
 * Wrappers for Eigen functions that preserve its lazy evaluation.
 */
#pragma once

namespace bi {
  namespace func {

template<class EigenType>
auto norm_(const EigenType& o) {
  return o.norm();
}

template<class Type, class Frame>
auto norm_(const Array<Type,Frame>& o) {
  return norm_(o.toEigen());
}

template<class EigenType>
auto squaredNorm_(const EigenType& o) {
  return o.squaredNorm();
}

template<class Type, class Frame>
auto squaredNorm_(const Array<Type,Frame>& o) {
  return squaredNorm_(o.toEigen());
}

template<class EigenType>
auto determinant_(const EigenType& o) {
  return o.determinant();
}

template<class Type, class Frame>
auto determinant_(const Array<Type,Frame>& o) {
  return determinant_(o.toEigen());
}

template<class EigenType>
auto transpose_(const EigenType& o) {
  return o.transpose();
}

template<class Type, class Frame>
auto transpose_(const Array<Type,Frame>& o) {
  return transpose_(o.toEigen());
}

template<class EigenType>
auto inverse_(const EigenType& o) {
  return o.inverse();
}

template<class Type, class Frame>
auto inverse_(const Array<Type,Frame>& o) {
  return inverse_(o.toEigen());
}

  }
}
