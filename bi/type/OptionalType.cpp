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

bool bi::OptionalType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::OptionalType::definitely(const AliasType& o) const {
  assert(o.target);
  return definitely(*o.target->base);
}

bool bi::OptionalType::definitely(const GenericType& o) const {
  assert(o.target);
  return definitely(*o.target->type);
}

bool bi::OptionalType::definitely(const OptionalType& o) const {
  return single->definitely(*o.single);
}

bool bi::OptionalType::definitely(const AnyType& o) const {
  return true;
}

bi::Type* bi::OptionalType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::OptionalType::common(const AliasType& o) const {
  assert(o.target);
  return common(*o.target->base);
}

bi::Type* bi::OptionalType::common(const GenericType& o) const {
  assert(o.target);
  return common(*o.target->type);
}

bi::Type* bi::OptionalType::common(const OptionalType& o) const {
  auto single1 = single->common(*o.single);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::OptionalType::common(const AnyType& o) const {
  return new AnyType();
}
