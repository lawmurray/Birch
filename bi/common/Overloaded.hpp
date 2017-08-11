/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/primitive/poset.hpp"
#include "bi/primitive/definitely.hpp"
#include "bi/primitive/possibly.hpp"

#include <map>

namespace bi {
/**
 * Overloadable object. Groups all overloads into one object.
 *
 * @ingroup compiler_common
 */
template<class ObjectType>
class Overloaded : public Named {
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
   * Iterators.
   */
  auto begin() const {
    return overloads.begin();
  }
  auto end() const {
    return overloads.end();
  }

  /**
   * Overloads.
   */
  std::list<ObjectType*> overloads;

  /**
   * Overload parameter types.
   */
  poset<Type*,definitely> params;

  /**
   * Map from parameter types to return types.
   */
  std::map<Type*,Type*> returns;
};
}
