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
 * @ingroup compiler_type
 */
class FiberType: public Type, public Single<Type> {
public:
  /**
   * Constructor.
   *
   * @param single Yield type.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  FiberType(Type* single, Location* loc = nullptr, const bool assignable =
      false);

  /**
   * Destructor.
   */
  virtual ~FiberType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isFiber() const;
  virtual Type* unwrap() const;

  using Type::definitely;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const AliasType& o) const;
  virtual bool definitely(const GenericType& o) const;
  virtual bool definitely(const FiberType& o) const;
  virtual bool definitely(const OptionalType& o) const;
};
}
