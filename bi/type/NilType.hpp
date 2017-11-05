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
   */
  NilType(Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~NilType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::common;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const NilType& o) const;
  virtual bool definitely(const OptionalType& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const NilType& o) const;
  virtual Type* common(const OptionalType& o) const;
};
}
