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
