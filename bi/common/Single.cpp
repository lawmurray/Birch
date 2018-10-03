/**
 * @file
 */
#include "bi/common/Single.hpp"

template<class T>
bi::Single<T>::Single(T* single) :
    single(single) {
  /* pre-conditions */
  assert(single);

  //
}

template<class T>
bi::Single<T>::~Single() {
  //
}

/*
 * Explicit instantiations.
 */
template class bi::Single<bi::Expression>;
template class bi::Single<bi::Statement>;
template class bi::Single<bi::Type>;
