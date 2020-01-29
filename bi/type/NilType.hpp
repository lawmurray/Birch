/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

namespace bi {
/**
 * Type of @c nil literal.
 *
 * @ingroup type
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

  using Type::isConvertible;
  using Type::isAssignable;

  virtual bool dispatchIsConvertible(const Type& o) const;
  virtual bool isConvertible(const NilType& o) const;
  virtual bool isConvertible(const OptionalType& o) const;

  virtual bool dispatchIsAssignable(const Type& o) const;
  virtual bool isAssignable(const NilType& o) const;
  virtual bool isAssignable(const OptionalType& o) const;
};
}
