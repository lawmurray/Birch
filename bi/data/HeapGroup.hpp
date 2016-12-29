/**
 * @file
 */
#pragma once

#include "bi/data/PrimitiveValue.hpp"
#include "bi/data/Frame.hpp"

namespace bi {
/**
 * Group for values on the heap.
 *
 * @ingroup library
 */
class HeapGroup {
public:
  template<class Value, class Frame = EmptyFrame>
  void create(PrimitiveValue<Value,HeapGroup>& value, const Frame& frame =
      EmptyFrame(), const char* name = nullptr);

  template<class Value, class Frame = EmptyFrame>
  void release(PrimitiveValue<Value,HeapGroup>& value, const Frame& frame =
      EmptyFrame());

private:
  /**
   * Number of bytes to which to align.
   */
  static const int_t ALIGNMENT = 32;
};
}

#include "bi/data/HeapPrimitiveValue.hpp"
#include "bi/exception/MemoryException.hpp"

#include <cstdlib>

template<class Value, class Frame>
void bi::HeapGroup::create(PrimitiveValue<Value,HeapGroup>& value,
    const Frame& frame, const char* name) {
  int err = posix_memalign((void**)&value.ptr, ALIGNMENT,
      frame.lead * sizeof(Value));
  if (err != 0) {
    throw MemoryException("Aligned memory allocation failed.");
  }
}

template<class Value, class Frame>
void bi::HeapGroup::release(PrimitiveValue<Value,HeapGroup>& value,
    const Frame& frame) {
  free(value.ptr);
}
