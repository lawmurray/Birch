/**
 * @file
 */
#include "src/type/TypeOf.hpp"

#include "src/visitor/all.hpp"

birch::TypeOf::TypeOf(Expression* single, Location* loc) :
    Type(loc),
    Single<Expression>(single) {
  //
}

birch::TypeOf::~TypeOf() {
  //
}

birch::Type* birch::TypeOf::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Type* birch::TypeOf::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::TypeOf::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
