/**
 * @file
 */
#pragma once

#include "bi/data/HeapGroup.hpp"
#include "bi/data/StackPrimitiveValue.hpp"

namespace bi {
/**
 * Value for HeapGroup.
 *
 * @ingroup library
 *
 * @tparam Type Primitive type.
 */
template<class Type>
class PrimitiveValue<Type,HeapGroup> {
public:
  typedef HeapGroup group_type;

  /**
   * Constructor.
   */
  template<class Frame = EmptyFrame>
  PrimitiveValue(const Frame& frame = EmptyFrame(),
      const char* name = nullptr, const HeapGroup& group = HeapGroup());

  /**
   * Constructor from stack.
   */
  PrimitiveValue(PrimitiveValue<Type,StackGroup>& value,
      const HeapGroup& group = HeapGroup());

  /**
   * View constructor.
   */
  template<class Frame, class View>
  PrimitiveValue(const PrimitiveValue<Type,HeapGroup>& o, const Frame& frame,
      const View& view);

  /**
   * Shallow copy constructor.
   */
  PrimitiveValue(const PrimitiveValue<Type,HeapGroup>& o);

  /**
   * Move constructor.
   */
  PrimitiveValue(PrimitiveValue<Type,HeapGroup> && o);

  /**
   * Destructor.
   */
  ~PrimitiveValue();

  /**
   * Assignment.
   */
  PrimitiveValue<Type,HeapGroup>& operator=(
      const PrimitiveValue<Type,HeapGroup>& o);

  /**
   * Generic assignment.
   */
  template<class Group1>
  PrimitiveValue<Type,HeapGroup>& operator=(
      const PrimitiveValue<Type,Group1>& o);

  /**
   * Value assignment.
   */
  PrimitiveValue<Type,HeapGroup>& operator=(const Type& o);

  /**
   * Value type conversion.
   */
  operator Type&();

  /**
   * Value type conversion.
   */
  operator const Type&() const;

  /**
   * Underlying buffer.
   */
  Type* ptr;

  /**
   * Group.
   */
  HeapGroup group;

  /**
   * Do we own the underlying buffer?
   */
  bool own;
};
}

#include "bi/data/copy.hpp"

template<class Type>
template<class Frame>
bi::PrimitiveValue<Type,bi::HeapGroup>::PrimitiveValue(const Frame& frame,
    const char* name, const HeapGroup& group) :
    group(group),
    own(true) {
  this->group.create(*this, frame, name);
}

template<class Type>
template<class Frame, class View>
bi::PrimitiveValue<Type,bi::HeapGroup>::PrimitiveValue(
    const PrimitiveValue<Type,HeapGroup>& o, const Frame& frame,
    const View& view) :
    ptr(o.ptr + frame.serial(view)),
    group(o.group),
    own(false) {
  //
}

template<class Type>
template<class Group1>
bi::PrimitiveValue<Type,bi::HeapGroup>& bi::PrimitiveValue<Type,bi::HeapGroup>::operator=(
    const PrimitiveValue<Type,Group1>& o) {
  copy(*this, o);
  return *this;
}
