/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

namespace bi {
/**
 * Type of @c nil literal.
 *
 * @ingroup compiler_type
 */
class NilType: public Type {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  NilType(Location* loc = nullptr, const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~NilType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const NilType& o) const;
  virtual bool definitely(const OptionalType& o) const;
};
}
