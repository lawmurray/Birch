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

void birch::TupleType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool birch::TupleType::isTuple() const {
  return true;
}
