/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Fiber type.
 *
 * @ingroup type
 */
class FiberType: public Type, public Single<Type> {
public:
  /**
   * Constructor.
   *
   * @param single Yield type.
   * @param loc Location.
   */
  FiberType(Type* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~FiberType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isFiber() const;
  virtual Type* unwrap();
  virtual const Type* unwrap() const;

  using Type::isConvertible;
  using Type::isAssignable;
  using Type::common;

  virtual bool dispatchIsConvertible(const Type& o) const;
  virtual bool isConvertible(const GenericType& o) const;
  virtual bool isConvertible(const MemberType& o) const;
  virtual bool isConvertible(const FiberType& o) const;
  virtual bool isConvertible(const OptionalType& o) const;

  virtual bool dispatchIsAssignable(const Type& o) const;
  virtual bool isAssignable(const ClassType& o) const;
  virtual bool isAssignable(const GenericType& o) const;
  virtual bool isAssignable(const MemberType& o) const;
  virtual bool isAssignable(const FiberType& o) const;
  virtual bool isAssignable(const OptionalType& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const GenericType& o) const;
  virtual Type* common(const MemberType& o) const;
  virtual Type* common(const FiberType& o) const;
  virtual Type* common(const OptionalType& o) const;
};
}
