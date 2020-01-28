/**
 * @file
 */
#include "bi/type/FiberType.hpp"

#include "bi/visitor/all.hpp"

bi::FiberType::FiberType(Type* yieldType, Location* loc) :
    Type(loc),
    YieldTyped(yieldType) {
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

bool bi::FiberType::isConvertible(const FiberType& o) const {
  return yieldType->equals(*o.yieldType);
  // ^ C++ code generation cannot handle the ->definitely case
}

bool bi::FiberType::isConvertible(const OptionalType& o) const {
  return isConvertible(*o.single);
}

bool bi::FiberType::dispatchIsAssignable(const Type& o) const {
  return o.isAssignable(*this);
}

bool bi::FiberType::isAssignable(const ClassType& o) const {
  return o.getClass()->hasAssignment(this);
}

bool bi::FiberType::isAssignable(const GenericType& o) const {
  assert(o.target);
  return isAssignable(*o.target->type);
}

bool bi::FiberType::isAssignable(const MemberType& o) const {
  return isAssignable(*o.right);
}

bool bi::FiberType::isAssignable(const FiberType& o) const {
  return yieldType->equals(*o.yieldType);
  // ^ C++ code generation cannot handle the ->definitely case
}

bool bi::FiberType::isAssignable(const OptionalType& o) const {
  return isAssignable(*o.single);
}

bi::Type* bi::FiberType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::FiberType::common(const GenericType& o) const {
  assert(o.target);
  return common(*o.target->type);
}

bi::Type* bi::FiberType::common(const MemberType& o) const {
  return common(*o.right);
}

bi::Type* bi::FiberType::common(const FiberType& o) const {
  auto yieldType1 = yieldType->common(*o.yieldType);
  if (yieldType1) {
    return new FiberType(yieldType1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::FiberType::common(const OptionalType& o) const {
  auto yieldType1 = common(*o.single);
  if (yieldType1) {
    return new OptionalType(yieldType1);
  } else {
    return nullptr;
  }
}
