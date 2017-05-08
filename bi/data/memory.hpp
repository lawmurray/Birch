/**
 * @file
 */
#pragma once

#include "bi/data/Frame.hpp"

#include <memory>

namespace bi {
/**
 * Allocate memory.
 *
 * @param ptr
 * @param frame
 */
template<class Type, class Frame = EmptyFrame>
void create(Type** ptr, const Frame& frame);

/**
 * Free memory.
 */
template<class Type, class Frame = EmptyFrame>
void release(Type* ptr, const Frame& frame);

/**
 * Initialise memory.
 *
 * @param ptr Memory address to initialise.
 * @param args Constructor arguments.
 */
template<class Type, class... Args>
void construct(Type* ptr, Args... args);

/**
 * Initialise memory.
 *
 * @param ptr Memory address to initialise.
 * @param args Constructor arguments.
 */
template<class Type, class... Args>
void construct(std::shared_ptr<Type>* ptr, Args... args);

/**
 * Number of bytes to which to align.
 */
static const int_t ALIGNMENT = 128;
}

#include "bi/exception/MemoryException.hpp"

#include <memory>
#include <type_traits>
#include <cstdlib>

template<class Type, class Frame>
void bi::create(Type** ptr, const Frame& frame) {
  int err = posix_memalign((void**)ptr, ALIGNMENT,
      frame.volume() * sizeof(Type));
  if (err != 0) {
    throw MemoryException("Aligned memory allocation failed.");
  }
}

template<class Type, class Frame>
void bi::release(Type* ptr, const Frame& frame) {
  free((void*)ptr);
}

template<class Type, class... Args>
void bi::construct(Type* ptr, Args... args) {
  return new (ptr) Type(args...);
}

template<class Type, class... Args>
void bi::construct(std::shared_ptr<Type>* ptr, Args... args) {
  return new (ptr) std::shared_ptr<Type>(std::make_shared<Type>(args...));
}
