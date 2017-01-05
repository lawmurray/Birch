/**
 * @file
 */
#include "bi/common/Unary.hpp"

#include <cassert>

template<class T>
bi::Unary<T>::Unary(T* single) :
    single(single) {
  /* pre-conditions */
  assert(single);

  //
}

template<class T>
bi::Unary<T>::~Unary() {
  //
}

/*
 * Explicit template instantiations.
 */
template class bi::Unary<bi::Expression>;
template class bi::Unary<bi::Statement>;
template class bi::Unary<bi::Type>;
