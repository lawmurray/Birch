/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Couple.hpp"

namespace bi {
/**
 * Membership operator type expression.
 *
 * @ingroup expression
 */
class MemberType: public Type, public Couple<Type> {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param right Right operand.
   * @param loc Location.
   */
  MemberType(Type* left, Type* right, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~MemberType();

  virtual bool isValue() const;
  virtual bool isBasic() const;
  virtual bool isClass() const;
  virtual bool isWeak() const;
  virtual bool isArray() const;
  virtual bool isFunction() const;
  virtual bool isFiber() const;
  virtual bool isMember() const;

  virtual Type* unwrap();
  virtual const Type* unwrap() const;

  virtual int depth() const;

  virtual Basic* getBasic() const;
  virtual Class* getClass() const;

  virtual Type* canonical();
  virtual const Type* canonical() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual void resolveConstructor(Argumented* o);

  using Type::isConvertible;
  using Type::isAssignable;
  using Type::common;

  virtual bool dispatchIsConvertible(const Type& o) const;
  virtual bool isConvertible(const ArrayType& o) const;
  virtual bool isConvertible(const BasicType& o) const;
  virtual bool isConvertible(const ClassType& o) const;
  virtual bool isConvertible(const EmptyType& o) const;
  virtual bool isConvertible(const FiberType& o) const;
  virtual bool isConvertible(const FunctionType& o) const;
  virtual bool isConvertible(const GenericType& o) const;
  virtual bool isConvertible(const MemberType& o) const;
  virtual bool isConvertible(const OptionalType& o) const;
  virtual bool isConvertible(const TupleType& o) const;
  virtual bool isConvertible(const TypeList& o) const;

  virtual bool dispatchIsAssignable(const Type& o) const;
  virtual bool isAssignable(const ArrayType& o) const;
  virtual bool isAssignable(const BasicType& o) const;
  virtual bool isAssignable(const ClassType& o) const;
  virtual bool isAssignable(const EmptyType& o) const;
  virtual bool isAssignable(const FiberType& o) const;
  virtual bool isAssignable(const FunctionType& o) const;
  virtual bool isAssignable(const GenericType& o) const;
  virtual bool isAssignable(const MemberType& o) const;
  virtual bool isAssignable(const OptionalType& o) const;
  virtual bool isAssignable(const TupleType& o) const;
  virtual bool isAssignable(const TypeList& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const ArrayType& o) const;
  virtual Type* common(const BasicType& o) const;
  virtual Type* common(const ClassType& o) const;
  virtual Type* common(const EmptyType& o) const;
  virtual Type* common(const FiberType& o) const;
  virtual Type* common(const FunctionType& o) const;
  virtual Type* common(const GenericType& o) const;
  virtual Type* common(const MemberType& o) const;
  virtual Type* common(const OptionalType& o) const;
  virtual Type* common(const TupleType& o) const;
  virtual Type* common(const TypeList& o) const;
};
}
