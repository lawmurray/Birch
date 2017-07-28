/**
 * @file
 */
#pragma once

#include "bi/primitive/poset.hpp"
#include "bi/primitive/definitely.hpp"
#include "bi/primitive/possibly.hpp"

namespace bi {
template<class T> class Call;

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
  void resolve(Call<ObjectType>* o);

  /**
   * Declarations by partial order.
   */
  poset_type overloads;
};
}
