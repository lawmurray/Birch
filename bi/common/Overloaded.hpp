/**
 * @file
 */
#pragma once

#include "bi/common/Named.hpp"
#include "bi/primitive/poset.hpp"
#include "bi/primitive/definitely.hpp"
#include "bi/primitive/possibly.hpp"

namespace bi {
/**
 * Overloadable object. Groups all overloads into one object.
 *
 * @ingroup compiler_common
 *
 * @tparam ObjectType Type of objects.
 */
template<class ObjectType>
class Overloaded: public Named {
public:
  typedef poset<ObjectType*,definitely> poset_type;

  /**
   * Constructor.
   *
   * @param o Initial object.
   */
  Overloaded(ObjectType* o);

  /**
   * Does this contain the given object?
   *
   * @param o The object.
   */
  bool contains(ObjectType* o) const;

  /**
   * Add an object.
   *
   * @param o The object.
   */
  void add(ObjectType* o);

  /**
   * Import from another overloaded name.
   */
  void import(Overloaded<ObjectType>& o);

  /**
   * Resolve reference.
   *
   * @param ref The reference.
   */
  template<class ReferenceType>
  void resolve(ReferenceType* ref);

  /**
   * Declarations by partial order.
   */
  poset_type objects;
};
}

template<class ObjectType>
template<class ReferenceType>
void bi::Overloaded<ObjectType>::resolve(ReferenceType* ref) {
  objects.match(ref, ref->matches);
}
