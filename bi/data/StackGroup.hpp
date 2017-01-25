/**
 * @file
 */
#pragma once

#include "bi/data/PrimitiveValue.hpp"
#include "bi/data/Frame.hpp"

#include <cstdlib>
#include <vector>

namespace bi {
class HeapGroup;

/**
 * Group for values on the stack.
 *
 * @ingroup library
 */
class StackGroup {
public:
  typedef StackGroup child_group_type;
  typedef HeapGroup array_group_type;

  template<class Type, class Frame = EmptyFrame>
  void create(PrimitiveValue<Type,StackGroup>& value,
      const Frame& frame = EmptyFrame(), const char* name = nullptr);

  template<class Type, class Frame = EmptyFrame>
  void release(PrimitiveValue<Type,StackGroup>& value,
      const Frame& frame = EmptyFrame());
};
}

#include "bi/data/StackPrimitiveValue.hpp"

template<class Type, class Frame>
void bi::StackGroup::create(PrimitiveValue<Type,StackGroup>& value,
    const Frame& frame, const char* name) {
  /* pre-condition */
  assert(frame.count() == 0);
}

template<class Type, class Frame>
void bi::StackGroup::release(PrimitiveValue<Type,StackGroup>& value,
    const Frame& frame) {
  /* pre-condition */
  assert(frame.count() == 0);
}
