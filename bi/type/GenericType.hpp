/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Reference.hpp"
#include "bi/expression/Generic.hpp"

namespace bi {
/**
 * Generic type.
 *
 * @ingroup type
 */
class GenericType: public Type, public Named, public Reference<Generic> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   * @param target Target.
   */
  GenericType(Name* name, Location* loc = nullptr, Generic* target = nullptr);

  /**
   * Constructor.
   *
   * @param target Target.
   */
  GenericType(Generic* target);

  /**
   * Destructor.
   */
  virtual ~GenericType();

  virtual bool isValue() const;
  virtual bool isBasic() const;
  virtual bool isClass() const;
  virtual bool isWeak() const;
  virtual bool isArray() const;
  virtual bool isFunction() const;
  virtual bool isFiber() const;
  virtual bool isGeneric() const;

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
  virtual bool isConvertible(const WeakType& o) const;
  virtual bool isConvertible(const SequenceType& o) const;
  virtual bool isConvertible(const TupleType& o) const;
  virtual bool isConvertible(const TypeList& o) const;

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
  virtual Type* common(const WeakType& o) const;
  virtual Type* common(const SequenceType& o) const;
  virtual Type* common(const TupleType& o) const;
  virtual Type* common(const TypeList& o) const;
};
}
