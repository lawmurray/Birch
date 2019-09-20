/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

namespace bi {
/**
 * Empty type.
 *
 * @ingroup type
 */
class EmptyType: public Type {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  EmptyType(Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~EmptyType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isValue() const;
  virtual bool isEmpty() const;

  using Type::isConvertible;
  using Type::isAssignable;
  using Type::common;

  virtual bool dispatchIsConvertible(const Type& o) const;
  virtual bool isConvertible(const EmptyType& o) const;
  virtual bool isConvertible(const GenericType& o) const;

  virtual bool dispatchIsAssignable(const Type& o) const;
  virtual bool isAssignable(const EmptyType& o) const;
  virtual bool isAssignable(const GenericType& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const EmptyType& o) const;
};
}
