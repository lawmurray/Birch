/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/primitive/poset.hpp"
#include "bi/primitive/definitely.hpp"
#include "bi/primitive/possibly.hpp"

namespace bi {
/**
 * Overloaded type. Typically used for the type of first-class functions.
 *
 * @ingroup compiler_common
 */
class OverloadedType: public Type {
public:
  /**
   * Constructor.
   *
   * @param o First overload.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  OverloadedType(Type* o, Location* loc = nullptr,
      const bool assignable = false);

  /**
   * Constructor.
   *
   * @param overloads Overloads.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  OverloadedType(const poset<Type*,bi::definitely>& overloads,
      Location* loc = nullptr, const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~OverloadedType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Does this contain the given overload?
   *
   * @param o The overload.
   */
  bool contains(Type* o) const;

  /**
   * Add an overload.
   *
   * @param o The overload.
   */
  void add(Type* o);

  virtual bool isOverloaded() const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const OverloadedType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const OverloadedType& o) const;

  /**
   * Declarations by partial order.
   */
  poset<Type*,bi::definitely> overloads;
};
}
