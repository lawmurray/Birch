/**
 * @file
 */
#pragma once

#include "bi/data/RefGroup.hpp"
#include "bi/data/HeapPrimitiveValue.hpp"
#include "bi/data/StackPrimitiveValue.hpp"

namespace bi {
/**
 * Value for RefGroup.
 *
 * @ingroup library
 *
 * @tparam Type Primitive type.
 */
template<class Type>
class PrimitiveValue<Type,RefGroup> {
public:
  typedef RefGroup group_type;

  template<class Group>
  using regroup_type = PrimitiveValue<Type,Group>;

  /**
   * Constructor from value.
   */
  PrimitiveValue(const Type& o, const EmptyFrame& frame = EmptyFrame(),
      const char* name = nullptr, const RefGroup& group = RefGroup());

  /**
   * Constructor from stack.
   */
  PrimitiveValue(const PrimitiveValue<Type,StackGroup>& o,
      const EmptyFrame& frame = EmptyFrame(), const char* name = nullptr,
      const RefGroup& group = RefGroup());

  /**
   * Constructor from heap.
   */
  PrimitiveValue(const PrimitiveValue<Type,HeapGroup>& o,
      const EmptyFrame& frame = EmptyFrame(), const char* name = nullptr,
      const RefGroup& group = RefGroup());

  /**
   * Assignment.
   */
  PrimitiveValue<Type,RefGroup>& operator=(
      const PrimitiveValue<Type,RefGroup>& o);

  /**
   * Generic assignment.
   */
  template<class Type1, class Group1>
  PrimitiveValue<Type,RefGroup>& operator=(
      const PrimitiveValue<Type1,Group1>& o);

  /**
   * Value assignment.
   */
  PrimitiveValue<Type,RefGroup>& operator=(const Type& o);

  /**
   * Basic type conversion.
   */
  operator Type&();

  /**
   * Basic type conversion.
   */
  operator const Type&() const;

  /**
   * Underlying value.
   */
  Type& value;

  /**
   * Group.
   */
  RefGroup group;
};
}

#include "bi/data/copy.hpp"

template<class Type>
template<class Type1, class Group1>
bi::PrimitiveValue<Type,bi::RefGroup>& bi::PrimitiveValue<Type,bi::RefGroup>::operator=(
    const PrimitiveValue<Type1,Group1>& o) {
  copy(*this, o);
  return *this;
}
