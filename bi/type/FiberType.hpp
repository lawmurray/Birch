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
  class FiberType: public Type, public YieldTyped, public ReturnTyped {
  public:
    /**
     * Constructor.
     *
     * @param yieldType Yield type.
     * @param returnType Return type.
     * @param loc Location.
     */
    FiberType(Type* yieldType, Type* returnType, Location* loc = nullptr);

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

    virtual bool dispatchIsConvertible(const Type& o) const;
    virtual bool isConvertible(const GenericType& o) const;
    virtual bool isConvertible(const MemberType& o) const;
    virtual bool isConvertible(const ArrayType& o) const;
    virtual bool isConvertible(const BasicType& o) const;
    virtual bool isConvertible(const ClassType& o) const;
    virtual bool isConvertible(const FiberType& o) const;
    virtual bool isConvertible(const FunctionType& o) const;
    virtual bool isConvertible(const OptionalType& o) const;
    virtual bool isConvertible(const TupleType& o) const;

    virtual bool dispatchIsAssignable(const Type& o) const;
    virtual bool isAssignable(const GenericType& o) const;
    virtual bool isAssignable(const MemberType& o) const;
    virtual bool isAssignable(const ArrayType& o) const;
    virtual bool isAssignable(const BasicType& o) const;
    virtual bool isAssignable(const ClassType& o) const;
    virtual bool isAssignable(const FiberType& o) const;
    virtual bool isAssignable(const FunctionType& o) const;
    virtual bool isAssignable(const OptionalType& o) const;
    virtual bool isAssignable(const TupleType& o) const;
  };
}
