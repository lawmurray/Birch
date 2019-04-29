/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Optional type.
 *
 * @ingroup type
 */
class OptionalType: public Type, public Single<Type> {
public:
  /**
   * Constructor.
   *
   * @param single Type.
   * @param loc Location.
   */
  OptionalType(Type* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~OptionalType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isValue() const;
  virtual bool isOptional() const;
  virtual Type* unwrap();
  virtual const Type* unwrap() const;

  using Type::isConvertible;
  using Type::common;

  virtual bool dispatchIsConvertible(const Type& o) const;
  virtual bool isConvertible(const GenericType& o) const;
  virtual bool isConvertible(const MemberType& o) const;
  virtual bool isConvertible(const OptionalType& o) const;
  virtual bool isConvertible(const WeakType& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const ArrayType& o) const;
  virtual Type* common(const BasicType& o) const;
  virtual Type* common(const BinaryType& o) const;
  virtual Type* common(const ClassType& o) const;
  virtual Type* common(const FiberType& o) const;
  virtual Type* common(const FunctionType& o) const;
  virtual Type* common(const GenericType& o) const;
  virtual Type* common(const MemberType& o) const;
  virtual Type* common(const NilType& o) const;
  virtual Type* common(const OptionalType& o) const;
  virtual Type* common(const WeakType& o) const;
  virtual Type* common(const SequenceType& o) const;
  virtual Type* common(const TupleType& o) const;
  virtual Type* common(const UnknownType& o) const;
  virtual Type* common(const TypeList& o) const;
};
}
