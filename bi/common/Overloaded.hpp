/**
 * @file
 */
#pragma once

#include "bi/expression/OverloadedCall.hpp"
#include "bi/primitive/poset.hpp"
#include "bi/primitive/definitely.hpp"
#include "bi/primitive/possibly.hpp"

namespace bi {
/**
 * Overloadable object. Groups all overloads into one object.
 *
 * @ingroup compiler_common
 */
template<class ObjectType>
class Overloaded {
public:
  typedef poset<ObjectType*,definitely> poset_type;

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
   * Resolve call.
   *
   * @param o The call.
   */
  void resolve(OverloadedCall<ObjectType>* o);

  /**
   * Declarations by partial order.
   */
  poset_type overloads;
};
}
