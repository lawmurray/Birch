/**
 * @file
 */
#include "src/type/TypeOf.hpp"

#include "src/visitor/all.hpp"

birch::TypeOf::TypeOf(Location* loc) :
    Type(loc) {
  //
}

bool birch::TypeOf::isTypeOf() const {
  return true;
}

birch::Type* birch::TypeOf::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::TypeOf::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
