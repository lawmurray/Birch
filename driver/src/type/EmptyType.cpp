/**
 * @file
 */
#include "src/type/EmptyType.hpp"

#include "src/visitor/all.hpp"

birch::EmptyType::EmptyType(Location* loc) :
    Type(loc) {
  //
}

void birch::EmptyType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool birch::EmptyType::isEmpty() const {
  return true;
}
