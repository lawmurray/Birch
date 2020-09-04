/**
 * @file
 */
#include "src/common/Single.hpp"

template<class T>
birch::Single<T>::Single(T* single) :
    single(single) {
  /* pre-conditions */
  assert(single);

  //
}

template<class T>
birch::Single<T>::~Single() {
  //
}

template class birch::Single<birch::Expression>;
template class birch::Single<birch::Statement>;
template class birch::Single<birch::Type>;
