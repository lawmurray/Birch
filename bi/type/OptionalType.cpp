/**
 * @file
 */
#include "bi/type/OptionalType.hpp"

#include "bi/visitor/all.hpp"

bi::OptionalType::OptionalType(Type* single, Location* loc) :
    Type(loc),
    Single<Type>(single) {
  //
}

bi::OptionalType::~OptionalType() {
  //
}

bi::Type* bi::OptionalType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::OptionalType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::OptionalType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::OptionalType::isOptional() const {
  return true;
}

const bi::Type* bi::OptionalType::unwrap() const {
  return single;
}

bi::Type* bi::OptionalType::unwrap() {
  return single;
}

bool bi::OptionalType::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bool bi::OptionalType::isConvertible(const GenericType& o) const {
  assert(o.target);
  return isConvertible(*o.target->type);
}

bool bi::OptionalType::isConvertible(const MemberType& o) const {
  return isConvertible(*o.right);
}

bool bi::OptionalType::isConvertible(const OptionalType& o) const {
  return single->isConvertible(*o.single);
}

bool bi::OptionalType::dispatchIsAssignable(const Type& o) const {
  return o.isAssignable(*this);
}

bool bi::OptionalType::isAssignable(const ClassType& o) const {
  return o.getClass()->hasAssignment(this);
}

bool bi::OptionalType::isAssignable(const GenericType& o) const {
  assert(o.target);
  return isAssignable(*o.target->type);
}

bool bi::OptionalType::isAssignable(const MemberType& o) const {
  return isAssignable(*o.right);
}

bool bi::OptionalType::isAssignable(const OptionalType& o) const {
  return single->isAssignable(*o.single);
}
