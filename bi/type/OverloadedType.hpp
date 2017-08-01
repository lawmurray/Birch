/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/primitive/poset.hpp"
#include "bi/primitive/definitely.hpp"
#include "bi/primitive/possibly.hpp"

#include <map>

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
   * @param params Overload parameter types.
   * @param returns Map from parameter to return types.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  OverloadedType(const poset<Type*,bi::definitely>& params,
      const std::map<Type*,Type*>& returns, Location* loc = nullptr,
      const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~OverloadedType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isOverloaded() const;
  virtual Type* resolve(Type* args);

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const OverloadedType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const OverloadedType& o) const;

  poset<Type*,bi::definitely> params;
  std::map<Type*,Type*> returns;
};
}
