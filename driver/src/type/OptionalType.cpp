/**
 * @file
 */
#include "src/type/OptionalType.hpp"

#include "src/visitor/all.hpp"

birch::OptionalType::OptionalType(Type* single, Location* loc) :
    Type(loc),
    Single<Type>(single) {
  //
}

birch::OptionalType::~OptionalType() {
  //
}

birch::Type* birch::OptionalType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Type* birch::OptionalType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::OptionalType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool birch::OptionalType::isOptional() const {
  return true;
}

bool birch::OptionalType::isValue() const {
  return single->isValue();
}

const birch::Type* birch::OptionalType::unwrap() const {
  return single;
}

birch::Type* birch::OptionalType::unwrap() {
  return single;
}
