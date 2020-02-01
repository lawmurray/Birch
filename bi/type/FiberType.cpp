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
