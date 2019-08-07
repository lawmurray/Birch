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
  using Type::common;

  virtual bool dispatchIsConvertible(const Type& o) const;
  virtual bool isConvertible(const NilType& o) const;
  virtual bool isConvertible(const OptionalType& o) const;
  virtual bool isConvertible(const WeakType& o) const;

  virtual bool dispatchIsAssignable(const Type& o) const;
  virtual bool isAssignable(const NilType& o) const;
  virtual bool isAssignable(const OptionalType& o) const;
  virtual bool isAssignable(const WeakType& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const NilType& o) const;
  virtual Type* common(const OptionalType& o) const;
  virtual Type* common(const WeakType& o) const;
};
}
