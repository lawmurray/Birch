/**
 * @file
 */
#pragma once

#include "bi/lib/Frame.hpp"
#include "bi/lib/reloc_ptr.hpp"

namespace bi {
/**
 * Allocate memory for non-pointer type.
 *
 * @tparam Type Element type.
 *
 * @param[out] ptr Pointer to start of allocated buffer.
 * @param size Number of bytes to allocate.
 */
template<class Type>
void create(reloc_ptr<Type>& ptr, const size_t size);

/**
 * Allocate memory for pointer type.
 *
 * @tparam Type Element type.
 *
 * @param[out] ptr Pointer to start of allocated buffer.
 * @param size Number of bytes to allocate.
 */
template<class Type>
void create(reloc_ptr<reloc_ptr<Type>>& ptr, const size_t size);

/**
 * Initialise memory.
 *
 * @param ptr Memory address to initialise.
 * @param args Constructor arguments.
 */
template<class Type, class... Args>
void construct(Type& o, Args... args);

/**
 * Initialise memory.
 *
 * @param ptr Memory address to initialise.
 * @param args Constructor arguments.
 */
template<class Type, class... Args>
void construct(reloc_ptr<Type>& o, Args... args);
}

#include <gc.h>

template<class Type>
void bi::create(reloc_ptr<Type>& ptr, const size_t size) {
  ptr = static_cast<Type*>(GC_MALLOC_ATOMIC(size));
  // ^ buffer cannot itself contain pointers, so GC_MALLOC_ATOMIC can be used
  //   to prevent the garbage collector sweeping through it
}

template<class Type>
void bi::create(reloc_ptr<reloc_ptr<Type>>& ptr, const size_t size) {
  ptr = static_cast<reloc_ptr<Type>*>(GC_MALLOC(size));
}

template<class Type, class... Args>
void bi::construct(Type& o, Args... args) {
  return new (&o) Type(args...);
}

template<class Type, class... Args>
void bi::construct(reloc_ptr<Type>& o, Args... args) {
  return new (&o) reloc_ptr<Type>(new (GC_MALLOC(sizeof(Type))) Type(args...));
}
