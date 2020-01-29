/**
 * @file
 */
#include "bi/type/FiberType.hpp"

#include "bi/visitor/all.hpp"

bi::FiberType::FiberType(Type* yieldType, Type* returnType, Location* loc) :
    Type(loc),
    YieldTyped(yieldType),
    ReturnTyped(returnType) {
  //
}

bi::FiberType::~FiberType() {
  //
}

bi::Type* bi::FiberType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::FiberType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::FiberType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::FiberType::isFiber() const {
  return true;
}

bi::Type* bi::FiberType::unwrap() {
  return yieldType;
}

const bi::Type* bi::FiberType::unwrap() const {
  return yieldType;
}

bool bi::FiberType::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bool bi::FiberType::isConvertible(const GenericType& o) const {
  assert(o.target);
  return isConvertible(*o.target->type);
}

bool bi::FiberType::isConvertible(const MemberType& o) const {
  return isConvertible(*o.right);
}

bool bi::FiberType::isConvertible(const ArrayType& o) const {
  return returnType->isConvertible(o);
}

bool bi::FiberType::isConvertible(const BasicType& o) const {
  return returnType->isConvertible(o);
}

bool bi::FiberType::isConvertible(const ClassType& o) const {
  return returnType->isConvertible(o);
}

bool bi::FiberType::isConvertible(const FiberType& o) const {
  return returnType->isConvertible(o) || (yieldType->equals(*o.yieldType) &&
      returnType->equals(*o.returnType));
}

bool bi::FiberType::isConvertible(const FunctionType& o) const {
  return returnType->isConvertible(o);
}

bool bi::FiberType::isConvertible(const OptionalType& o) const {
  return returnType->isConvertible(o) || isConvertible(*o.single);
}

bool bi::FiberType::isConvertible(const TupleType& o) const {
  return returnType->isConvertible(o);
}

bool bi::FiberType::dispatchIsAssignable(const Type& o) const {
  return o.isAssignable(*this);
}

bool bi::FiberType::isAssignable(const GenericType& o) const {
  assert(o.target);
  return returnType->isAssignable(o) || isAssignable(*o.target->type);
}

bool bi::FiberType::isAssignable(const MemberType& o) const {
  return returnType->isAssignable(o) || isAssignable(*o.right);
}

bool bi::FiberType::isAssignable(const ArrayType& o) const {
  return returnType->isAssignable(o);
}

bool bi::FiberType::isAssignable(const BasicType& o) const {
  return returnType->isAssignable(o);
}

bool bi::FiberType::isAssignable(const ClassType& o) const {
  return returnType->isAssignable(o) || o.getClass()->hasAssignment(this);
}

bool bi::FiberType::isAssignable(const FiberType& o) const {
  return returnType->isAssignable(o) || (yieldType->equals(*o.yieldType) &&
      returnType->equals(*o.returnType));
}

bool bi::FiberType::isAssignable(const FunctionType& o) const {
  return returnType->isAssignable(o);
}

bool bi::FiberType::isAssignable(const OptionalType& o) const {
  return returnType->isAssignable(o) || isAssignable(*o.single);
}

bool bi::FiberType::isAssignable(const TupleType& o) const {
  return returnType->isAssignable(o);
}
