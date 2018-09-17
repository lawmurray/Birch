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

  virtual bool isValue() const;
  virtual bool isFiber() const;
  virtual Type* unwrap();
  virtual const Type* unwrap() const;

  using Type::definitely;
  using Type::common;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const GenericType& o) const;
  virtual bool definitely(const MemberType& o) const;
  virtual bool definitely(const FiberType& o) const;
  virtual bool definitely(const OptionalType& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const GenericType& o) const;
  virtual Type* common(const MemberType& o) const;
  virtual Type* common(const FiberType& o) const;
  virtual Type* common(const OptionalType& o) const;
};
}
