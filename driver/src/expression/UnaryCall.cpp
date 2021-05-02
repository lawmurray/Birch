/**
 * @file
 */
#include "src/expression/UnaryCall.hpp"

#include "src/visitor/all.hpp"

birch::UnaryCall::UnaryCall(Name* name, Expression* single, Location* loc) :
    Expression(loc),
    Named(name),
    Single<Expression>(single) {
  //
}

birch::Expression* birch::UnaryCall::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::UnaryCall::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
