/**
 * @file
 */
#include "src/type/TupleType.hpp"

#include "src/visitor/all.hpp"

birch::TupleType::TupleType(Type* single, Location* loc) :
    Type(loc),
    Single<Type>(single) {
  //
}

birch::TupleType::~TupleType() {
  //
}

birch::Type* birch::TupleType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Type* birch::TupleType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::TupleType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool birch::TupleType::isTuple() const {
  return true;
}

bool birch::TupleType::isValue() const {
  return single->isValue();
}
