/**
 * @file
 */
#pragma once

#include "bi/data/Frame.hpp"

namespace bi {
/**
 * Allocate memory.
 *
 * @param ptr
 * @param frame
 */
template<class Type, class Frame = EmptyFrame>
void create(Type* ptr, const Frame& frame);

/**
 * Initialise memory.
 */
template<class Type, class Frame = EmptyFrame>
void fill(Type* ptr, const Frame& frame, const Type& init = Type());

/**
 * Free memory.
 */
template<class Type, class Frame = EmptyFrame>
void release(Type* ptr, const Frame& frame);

/**
 * Number of bytes to which to align.
 */
static const int_t ALIGNMENT = 128;
}

#include "bi/exception/MemoryException.hpp"

#include <cstdlib>

template<class Type, class Frame>
void bi::create(Type* ptr, const Frame& frame) {
  int err = posix_memalign((void**)&ptr, ALIGNMENT,
      frame.volume() * sizeof(Type));
  if (err != 0) {
    throw MemoryException("Aligned memory allocation failed.");
  }
}

template<class Type, class Frame>
void bi::fill(Type* ptr, const Frame& frame, const Type& init) {
  std::uninitialized_fill_n(ptr, frame.volume(), init);
}

template<class Type, class Frame>
void bi::release(Type* ptr, const Frame& frame) {
  free(ptr);
}
