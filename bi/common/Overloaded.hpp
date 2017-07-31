/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/primitive/unique_ptr.hpp"

#include <list>

namespace bi {
/**
 * Overloadable object. Groups all overloads into one object.
 *
 * @ingroup compiler_common
 */
template<class ObjectType>
class Overloaded {
public:
  /**
   * Constructor.
   *
   * @param o First overload.
   */
  Overloaded(ObjectType* o);

  /**
   * Does this contain the given overload?
   *
   * @param o The overload.
   */
  bool contains(ObjectType* o) const;

  /**
   * Add an overload.
   *
   * @param o The overload.
   */
  void add(ObjectType* o);

  /**
   * Type. Should be of type OverloadedType.
   */
  unique_ptr<Type> type;

  /**
   * Overloads.
   *
   * std::list is preferred to std::set to maintain declaration order of
   * overloads for error messages.
   */
  std::list<ObjectType*> overloads;
};
}
