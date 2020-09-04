/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"
#include "src/common/YieldTyped.hpp"
#include "src/common/ReturnTyped.hpp"

namespace birch {
  /**
   * Fiber type.
   *
   * @ingroup type
   */
  class FiberType: public Type, public ReturnTyped, public YieldTyped {
  public:
    /**
     * Constructor.
     *
     * @param returnType Return type.
     * @param yieldType Yield type.
     * @param loc Location.
     */
    FiberType(Type* returnType, Type* yieldType, Location* loc = nullptr);

    /**
     * Destructor.
     */
    virtual ~FiberType();

    virtual Type* accept(Cloner* visitor) const;
    virtual Type* accept(Modifier* visitor);
    virtual void accept(Visitor* visitor) const;

    virtual bool isFiber() const;
    virtual bool isValue() const;
  };
}
