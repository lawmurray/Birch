/**
 * @file
 */
#include "bi/common/Couple.hpp"

#include <cassert>

template<class T>
bi::Couple<T>::Couple(T* left, T* right) :
    left(left), right(right) {
  /* pre-conditions */
  assert(left);
  assert(right);

  //
}

template<class T>
bi::Couple<T>::~Couple() {
  //
}

/*
 * Explicit template instantiations.
 */
template class bi::Couple<bi::Expression>;
template class bi::Couple<bi::Type>;
