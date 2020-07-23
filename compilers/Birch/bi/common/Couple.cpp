/**
 * @file
 */
#include "bi/common/Couple.hpp"

#include "bi/expression/Expression.hpp"
#include "bi/type/Type.hpp"

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

template class bi::Couple<bi::Expression>;
template class bi::Couple<bi::Type>;
