/**
 * @file
 */
#include "bi/common/Binary.hpp"

#include <cassert>

template<class T>
bi::Binary<T>::Binary(T* left, T* right) :
    left(left), right(right) {
  /* pre-conditions */
  assert(left);
  assert(right);

  //
}

template<class T>
bi::Binary<T>::~Binary() {
  //
}

/*
 * Explicit template instantiations.
 */
template class bi::Binary<bi::Expression>;
template class bi::Binary<bi::Type>;
