/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Cast away the const-ness of a pointer. Usually used for the @c this
 * pointer to implement the read-write semantics of models.
 */
template<class T>
T* nonconst(const T* o) {
  return const_cast<T*>(o);
}
}
