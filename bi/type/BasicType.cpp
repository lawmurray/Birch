/**
 * @file
 */
#include "bi/type/BasicType.hpp"

#include "bi/visitor/all.hpp"

bi::BasicType::BasicType(Name* name, Location* loc, Basic* target) :
    Type(loc),
    Named(name),
    Reference<Basic>(target) {
  //
}

bi::BasicType::BasicType(Basic* target) :
    Named(target->name),
    Reference<Basic>(target) {
  //
}

bi::BasicType::~BasicType() {
  //
}

bi::Type* bi::BasicType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::BasicType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::BasicType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::BasicType::isBasic() const {
  return true;
}

bi::Basic* bi::BasicType::getBasic() const {
  return target;
}

bi::Type* bi::BasicType::canonical() {
  assert(target);
  if (target->alias) {
    return target->base->canonical();
  } else {
    return this;
  }
}

const bi::Type* bi::BasicType::canonical() const {
  assert(target);
  if (target->alias) {
    return target->base->canonical();
  } else {
    return this;
  }
}

bool bi::BasicType::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bool bi::BasicType::isConvertible(const GenericType& o) const {
  assert(o.target);
  return isConvertible(*o.target->type);
}

bool bi::BasicType::isConvertible(const MemberType& o) const {
  return isConvertible(*o.right);
}

bool bi::BasicType::isConvertible(const BasicType& o) const {
  assert(target);
  auto o1 = o.canonical();
  return target == o1->getBasic() || target->hasSuper(o1);
}

bool bi::BasicType::isConvertible(const OptionalType& o) const {
  return isConvertible(*o.single);
}

bool bi::BasicType::dispatchIsAssignable(const Type& o) const {
  return o.isAssignable(*this);
}

bool bi::BasicType::isAssignable(const ClassType& o) const {
  return o.getClass()->hasAssignment(this);
}

bool bi::BasicType::isAssignable(const GenericType& o) const {
  assert(o.target);
  return isAssignable(*o.target->type);
}

bool bi::BasicType::isAssignable(const MemberType& o) const {
  return isAssignable(*o.right);
}

bool bi::BasicType::isAssignable(const BasicType& o) const {
  assert(target);
  auto o1 = o.canonical();
  return target == o1->getBasic() || target->hasSuper(o1);
}

bool bi::BasicType::isAssignable(const OptionalType& o) const {
  return isAssignable(*o.single);
}

bi::Type* bi::BasicType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::BasicType::common(const GenericType& o) const {
  assert(o.target);
  return common(*o.target->type);
}

bi::Type* bi::BasicType::common(const MemberType& o) const {
  return common(*o.right);
}

bi::Type* bi::BasicType::common(const BasicType& o) const {
  assert(target);
  assert(o.target);
  if (target == o.target) {
    return new BasicType(target);
  } else if (target->hasSuper(&o)) {
    return new BasicType(o.target);
  } else if (o.target->hasSuper(this)) {
    return new BasicType(target);
  } else {
    return target->base->common(*o.target->base);
  }
}

bi::Type* bi::BasicType::common(const OptionalType& o) const {
  auto single1 = common(*o.single);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}
