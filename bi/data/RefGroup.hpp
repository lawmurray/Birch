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
 * Group for reference values.
 *
 * @ingroup library
 */
class RefGroup {
public:
  typedef RefGroup child_group_type;
  typedef HeapGroup array_group_type;

  template<class Type, class Frame = EmptyFrame>
  void create(PrimitiveValue<Type,RefGroup>& value,
      const Frame& frame = EmptyFrame(), const char* name = nullptr);

  template<class Type, class Frame = EmptyFrame>
  void release(PrimitiveValue<Type,RefGroup>& value,
      const Frame& frame = EmptyFrame());
};
}

#include "bi/data/RefPrimitiveValue.hpp"

template<class Type, class Frame>
void bi::RefGroup::create(PrimitiveValue<Type,RefGroup>& value,
    const Frame& frame, const char* name) {
  /* pre-condition */
  assert(frame.count() == 0);
}

template<class Type, class Frame>
void bi::RefGroup::release(PrimitiveValue<Type,RefGroup>& value,
    const Frame& frame) {
  /* pre-condition */
  assert(frame.count() == 0);
}
