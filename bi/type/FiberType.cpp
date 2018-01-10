/**
 * @file
 */
#include "bi/type/FiberType.hpp"

#include "bi/visitor/all.hpp"

bi::FiberType::FiberType(Type* single, Location* loc) :
    Type(loc),
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

bi::Type* bi::FiberType::unwrap() {
  return single;
}

const bi::Type* bi::FiberType::unwrap() const {
  return single;
}

bool bi::FiberType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
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

bool bi::FiberType::definitely(const AnyType& o) const {
  return true;
}

bi::Type* bi::FiberType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::FiberType::common(const GenericType& o) const {
  assert(o.target);
  return common(*o.target->type);
}

bi::Type* bi::FiberType::common(const FiberType& o) const {
  auto single1 = single->common(*o.single);
  if (single1) {
    return new FiberType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::FiberType::common(const OptionalType& o) const {
  auto single1 = common(*o.single);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::FiberType::common(const AnyType& o) const {
  return new AnyType();
}
