/**
 * @file
 */
#include "src/type/DeducedType.hpp"

#include "src/visitor/all.hpp"

birch::DeducedType::DeducedType(Location* loc) :
    Type(loc) {
  //
}

bool birch::DeducedType::isDeduced() const {
  return true;
}

void birch::DeducedType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
