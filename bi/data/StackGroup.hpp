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

  template<class Type>
  void create(PrimitiveValue<Type,StackGroup>& value,
      const EmptyFrame& frame = EmptyFrame(), const char* name = nullptr);

  template<class Type>
  void release(PrimitiveValue<Type,StackGroup>& value,
      const EmptyFrame& frame = EmptyFrame());
};
}

#include "bi/data/StackPrimitiveValue.hpp"

template<class Type>
void bi::StackGroup::create(PrimitiveValue<Type,StackGroup>& value,
    const EmptyFrame& frame, const char* name) {
  //
}

template<class Type>
void bi::StackGroup::release(PrimitiveValue<Type,StackGroup>& value,
    const EmptyFrame& frame) {
  //
}
