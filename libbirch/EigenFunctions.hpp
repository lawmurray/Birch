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
auto norm_(const EigenType& o) {
  return o.norm();
}

template<class Type, class Frame>
auto norm_(const Array<Type,Frame>& o) {
  return norm_(o.toEigen());
}

template<class EigenType>
auto dot_(const EigenType& o) {
  return o.squaredNorm();
}

template<class Type, class Frame>
auto dot_(const Array<Type,Frame>& o) {
  return dot_(o.toEigen());
}

template<class EigenType>
auto det_(const EigenType& o) {
  return o.determinant();
}

template<class Type, class Frame>
auto det_(const Array<Type,Frame>& o) {
  return det_(o.toEigen());
}

template<class EigenType>
auto trans_(const EigenType& o) {
  return o.transpose();
}

template<class Type, class Frame>
auto trans_(const Array<Type,Frame>& o) {
  return trans_(o.toEigen());
}

template<class EigenType>
auto inv_(const EigenType& o) {
  return o.inverse();
}

template<class Type, class Frame>
auto inv_(const Array<Type,Frame>& o) {
  return inv_(o.toEigen());
}

}
