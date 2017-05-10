/**
 * @file
 */
#pragma once

#include <memory>

namespace bi {
/**
 * Cast away the const-ness of a pointer. Usually used for the @c this
 * pointer to implement the read-write semantics of objects.
 */
template<class T>
T* cast(const T* o) {
  return const_cast<T*>(o);
}

/**
 * Cast away the const-ness of a reference. Usually used for passing rvalues
 * to assignment operator overloads requiring lvalue reference argument.
 */
template<class T>
T& cast(const T& o) {
  return const_cast<T&>(o);
}

/**
 * Simultaneously cast a shared pointer to a derived type and remove its
 * constness.
 */
template<class To, class From>
std::shared_ptr<To> cast(const std::shared_ptr<const From>& o) {
  return std::dynamic_pointer_cast<To>(std::const_pointer_cast<From>(o));
}
}
