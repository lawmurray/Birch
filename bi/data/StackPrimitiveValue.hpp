/**
 * @file
 */
#pragma once

#include "bi/data/StackGroup.hpp"

namespace bi {
/**
 * Value for StackGroup.
 *
 * @ingroup library
 *
 * @tparam Type Primitive type.
 */
template<class Type>
class PrimitiveValue<Type,StackGroup> {
public:
  typedef StackGroup group_type;

  template<class Group>
  using regroup_type = PrimitiveValue<Type,Group>;

  /**
   * Constructor.
   */
  PrimitiveValue(const EmptyFrame& frame = EmptyFrame(), const char* name =
      nullptr, const StackGroup& group = StackGroup());

  /**
   * Constructor from value.
   */
  PrimitiveValue(const Type& value, const EmptyFrame& frame = EmptyFrame(),
      const char* name = nullptr, const StackGroup& group = StackGroup());

  /**
   * Constructor from heap.
   */
  PrimitiveValue(const PrimitiveValue<Type,HeapGroup>& o,
      const EmptyFrame& frame = EmptyFrame(), const char* name = nullptr,
      const StackGroup& group = StackGroup());

  /**
   * Assignment.
   */
  PrimitiveValue<Type,StackGroup>& operator=(
      const PrimitiveValue<Type,StackGroup>& o);

  /**
   * Generic assignment.
   */
  template<class Type1, class Group1>
  PrimitiveValue<Type,StackGroup>& operator=(
      const PrimitiveValue<Type1,Group1>& o);

  /**
   * Value assignment.
   */
  PrimitiveValue<Type,StackGroup>& operator=(const Type& o);

  /**
   * Value type conversion.
   */
  operator Type&();

  /**
   * Value type conversion.
   */
  operator const Type&() const;

  /**
   * Underlying value.
   */
  Type value;

  /**
   * Group.
   */
  StackGroup group;
};
}

#include "bi/data/copy.hpp"

template<class Type>
template<class Type1, class Group1>
bi::PrimitiveValue<Type,bi::StackGroup>& bi::PrimitiveValue<Type,
    bi::StackGroup>::operator=(const PrimitiveValue<Type1,Group1>& o) {
  copy(*this, o);
  return *this;
}
