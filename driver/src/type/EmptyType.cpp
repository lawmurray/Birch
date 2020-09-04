/**
 * @file
 */
#include "src/type/EmptyType.hpp"

#include "src/visitor/all.hpp"

birch::EmptyType::EmptyType(Location* loc) :
    Type(loc) {
  //
}

birch::EmptyType::~EmptyType() {
  //
}

birch::Type* birch::EmptyType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Type* birch::EmptyType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::EmptyType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool birch::EmptyType::isEmpty() const {
  return true;
}

bool birch::EmptyType::isValue() const {
  return true;
}
