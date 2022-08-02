/**
 * @file
 */
#include "src/type/FutureType.hpp"

#include "src/visitor/all.hpp"

birch::FutureType::FutureType(Type* single, Location* loc) :
    Type(loc),
    Single<Type>(single) {
  //
}

void birch::FutureType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool birch::FutureType::isFuture() const {
  return true;
}

const birch::Type* birch::FutureType::unwrap() const {
  return single;
}

birch::Type* birch::FutureType::unwrap() {
  return single;
}
