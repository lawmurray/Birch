/**
 * @file
 */
#include "numbirch/memory.hpp"

void* numbirch::malloc(const size_t size) {
  return size == 0 ? nullptr : numbirch_mallocx(size, shared_flags);
}

void* numbirch::realloc(void* ptr, const size_t size) {
  if (size > 0) {
    return numbirch_rallocx(ptr, size, shared_flags);
  } else {
    free(ptr);
    return nullptr;
  }
}

void numbirch::free(void* ptr) {
  /// @todo Actually need to wait on the stream associated with the arena
  /// where this allocation was made, and only if its a different thread to
  /// this one, lest it is reused by the associated thread before this thread
  /// has finished any asynchronous work
  if (ptr) {
    numbirch_dallocx(ptr, shared_flags);
  }
}
