/**
 * @file
 */
#include "src/type/FiberType.hpp"

#include "src/visitor/all.hpp"

birch::FiberType::FiberType(Type* returnType, Type* yieldType, Location* loc) :
    Type(loc),
    ReturnTyped(returnType),
    YieldTyped(yieldType) {
  //
}

birch::FiberType::~FiberType() {
  //
}

birch::Type* birch::FiberType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Type* birch::FiberType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::FiberType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool birch::FiberType::isFiber() const {
  return true;
}

bool birch::FiberType::isValue() const {
  return false;
}
