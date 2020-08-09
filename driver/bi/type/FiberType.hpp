/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/YieldTyped.hpp"
#include "bi/common/ReturnTyped.hpp"

namespace bi {
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
