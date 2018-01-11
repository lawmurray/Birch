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

bool bi::BasicType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::BasicType::definitely(const GenericType& o) const {
  assert(o.target);
  return definitely(*o.target->type);
}

bool bi::BasicType::definitely(const BasicType& o) const {
  assert(target);
  auto o1 = o.canonical();
  return target == o1->getBasic() || target->hasSuper(o1);
}

bool bi::BasicType::definitely(const OptionalType& o) const {
  return definitely(*o.single);
}

bool bi::BasicType::definitely(const AnyType& o) const {
  return true;
}

bi::Type* bi::BasicType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::BasicType::common(const GenericType& o) const {
  assert(o.target);
  return common(*o.target->type);
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

bi::Type* bi::BasicType::common(const AnyType& o) const {
  return new AnyType();
}
