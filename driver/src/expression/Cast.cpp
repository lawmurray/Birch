/**
 * @file
 */
#include "src/expression/Cast.hpp"

#include "src/visitor/all.hpp"

birch::Cast::Cast(Type* returnType, Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single),
    ReturnTyped(returnType) {
  //
}

birch::Expression* birch::Cast::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Cast::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
