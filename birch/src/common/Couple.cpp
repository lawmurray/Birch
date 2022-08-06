/**
 * @file
 */
#include "src/common/Couple.hpp"

#include "src/expression/Expression.hpp"
#include "src/type/Type.hpp"

template<class T>
birch::Couple<T>::Couple(T* left, T* right) :
    left(left), right(right) {
  /* pre-conditions */
  assert(left);
  assert(right);

  //
}

template class birch::Couple<birch::Expression>;
template class birch::Couple<birch::Type>;
