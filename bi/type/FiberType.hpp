/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/ReturnTyped.hpp"

namespace bi {
/**
 * Fiber type.
 *
 * @ingroup compiler_type
 */
class FiberType: public Type, public ReturnTyped {
public:
  /**
   * Constructor.
   *
   * @param returnType Return type.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  FiberType(Type* returnType = new EmptyType(), Location* loc =
      nullptr, const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~FiberType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isCoroutine() const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const FiberType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const FiberType& o) const;
};
}
