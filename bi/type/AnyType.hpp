/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

namespace bi {
/**
 * Empty type.
 *
 * @ingroup birch_type
 */
class AnyType: public Type {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  AnyType(Location* loc =  nullptr);

  /**
   * Destructor.
   */
  virtual ~AnyType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isEmpty() const;

  using Type::definitely;
  using Type::common;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const AnyType& o) const;
  virtual bool definitely(const GenericType& o) const;
  virtual bool definitely(const OptionalType& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const AnyType& o) const;
  virtual Type* common(const GenericType& o) const;
  virtual Type* common(const OptionalType& o) const;
};
}
