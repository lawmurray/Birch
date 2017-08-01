/**
 * @file
 */
#include "bi/type/FiberType.hpp"

#include "bi/visitor/all.hpp"

bi::FiberType::FiberType(Type* returnType, Location* loc,
    const bool assignable) :
    Type(loc, assignable),
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

bool bi::FiberType::isCoroutine() const {
  return true;
}

bool bi::FiberType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::FiberType::definitely(const FiberType& o) const {
  return returnType->definitely(*o.returnType);
}

bool bi::FiberType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::FiberType::possibly(const FiberType& o) const {
  return returnType->possibly(*o.returnType);
}
