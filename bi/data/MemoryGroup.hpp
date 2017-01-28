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
class MemoryGroup {
public:
  template<class Value, class Frame = EmptyFrame>
  void create(PrimitiveValue<Value,MemoryGroup>& value, const Frame& frame =
      EmptyFrame(), const char* name = nullptr);

  template<class Value, class Frame = EmptyFrame>
  void create(PrimitiveValue<Value,MemoryGroup>& value, const Value& init,
      const Frame& frame = EmptyFrame(), const char* name = nullptr);

  template<class Value, class Frame = EmptyFrame>
  void release(PrimitiveValue<Value,MemoryGroup>& value, const Frame& frame =
      EmptyFrame());

private:
  /**
   * Number of bytes to which to align.
   */
  static const int_t ALIGNMENT = 32;
};
}

#include "bi/data/MemoryPrimitiveValue.hpp"
#include "bi/exception/MemoryException.hpp"

#include <cstdlib>

template<class Value, class Frame>
void bi::MemoryGroup::create(PrimitiveValue<Value,MemoryGroup>& value,
    const Frame& frame, const char* name) {
  int err = posix_memalign((void**)&value.ptr, ALIGNMENT,
      frame.lead * sizeof(Value));
  if (err != 0) {
    throw MemoryException("Aligned memory allocation failed.");
  }
}

template<class Value, class Frame>
void bi::MemoryGroup::create(PrimitiveValue<Value,MemoryGroup>& value,
    const Value& init, const Frame& frame, const char* name) {
  create(value, frame, name);
  std::uninitialized_fill_n(value.ptr, frame.lead, init);
}

template<class Value, class Frame>
void bi::MemoryGroup::release(PrimitiveValue<Value,MemoryGroup>& value,
    const Frame& frame) {
  free(value.ptr);
}
