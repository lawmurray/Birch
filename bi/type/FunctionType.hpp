/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/ReturnTyped.hpp"

namespace bi {
/**
 * Function type.
 *
 * @ingroup type
 */
class FunctionType: public Type, public ReturnTyped {
public:
  /**
   * Constructor.
   *
   * @param params Parameters type.
   * @param returnType Return type.
   * @param loc Location.
   */
  FunctionType(Type* params, Type* returnType, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~FunctionType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isFunction() const;

  using Type::isConvertible;
  using Type::isAssignable;
  using Type::common;

  virtual bool dispatchIsConvertible(const Type& o) const;
  virtual bool isConvertible(const GenericType& o) const;
  virtual bool isConvertible(const MemberType& o) const;
  virtual bool isConvertible(const FunctionType& o) const;
  virtual bool isConvertible(const OptionalType& o) const;

  virtual bool dispatchIsAssignable(const Type& o) const;
  virtual bool isAssignable(const ClassType& o) const;
  virtual bool isAssignable(const GenericType& o) const;
  virtual bool isAssignable(const MemberType& o) const;
  virtual bool isAssignable(const FunctionType& o) const;
  virtual bool isAssignable(const OptionalType& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const GenericType& o) const;
  virtual Type* common(const MemberType& o) const;
  virtual Type* common(const FunctionType& o) const;
  virtual Type* common(const OptionalType& o) const;

  /**
   * Parameters type.
   */
  Type* params;
};
}
