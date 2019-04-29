/**
 * @file
 */
#include "bi/type/TupleType.hpp"

#include "bi/visitor/all.hpp"

bi::TupleType::TupleType(Type* single, Location* loc) :
    Type(loc),
    Single<Type>(single) {
  //
}

bi::TupleType::~TupleType() {
  //
}

bool bi::TupleType::isValue() const {
  return single->isValue();
}

bi::Type* bi::TupleType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::TupleType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::TupleType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::TupleType::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bool bi::TupleType::isConvertible(const GenericType& o) const {
  assert(o.target);
  return isConvertible(*o.target->type);
}

bool bi::TupleType::isConvertible(const MemberType& o) const {
  return isConvertible(*o.right);
}

bool bi::TupleType::isConvertible(const OptionalType& o) const {
  return isConvertible(*o.single);
}

bool bi::TupleType::isConvertible(const TupleType& o) const {
  return single->isConvertible(*o.single);
}

bool bi::TupleType::dispatchIsAssignable(const Type& o) const {
  return o.isAssignable(*this);
}

bool bi::TupleType::isAssignable(const ClassType& o) const {
  return o.getClass()->hasAssignment(this);
}

bool bi::TupleType::isAssignable(const GenericType& o) const {
  assert(o.target);
  return isAssignable(*o.target->type);
}

bool bi::TupleType::isAssignable(const MemberType& o) const {
  return isAssignable(*o.right);
}

bool bi::TupleType::isAssignable(const OptionalType& o) const {
  return isAssignable(*o.single);
}

bool bi::TupleType::isAssignable(const TupleType& o) const {
  return single->isAssignable(*o.single);
}

bi::Type* bi::TupleType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::TupleType::common(const GenericType& o) const {
  assert(o.target);
  return common(*o.target->type);
}

bi::Type* bi::TupleType::common(const MemberType& o) const {
  return common(*o.right);
}

bi::Type* bi::TupleType::common(const OptionalType& o) const {
  auto single1 = common(*o.single);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::TupleType::common(const TupleType& o) const {
  auto single1 = single->common(*o.single);
  if (single1) {
    return new TupleType(single1);
  } else {
    return nullptr;
  }
}
