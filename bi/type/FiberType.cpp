/**
 * @file
 */
#include "bi/type/FiberType.hpp"

#include "bi/visitor/all.hpp"

bi::FiberType::FiberType(Type* single, Location* loc, const bool assignable) :
    Type(loc, assignable),
    Single<Type>(single) {
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

bi::Type* bi::FiberType::unwrap() const {
  return single;
}

bool bi::FiberType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::FiberType::definitely(const AliasType& o) const {
  assert(o.target);
  return definitely(*o.target->base);
}

bool bi::FiberType::definitely(const GenericType& o) const {
  assert(o.target);
  return definitely(*o.target->type);
}

bool bi::FiberType::definitely(const FiberType& o) const {
  return single->definitely(*o.single);
}

bool bi::FiberType::definitely(const OptionalType& o) const {
  return definitely(*o.single);
}

bool bi::FiberType::definitely(const EmptyType& o) const {
  return true;
}
