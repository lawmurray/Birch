/**
 * @file
 */
#include "bi/expression/UnaryCall.hpp"

#include "bi/visitor/all.hpp"

bi::UnaryCall::UnaryCall(Name* name, Expression* single, Location* loc) :
    Expression(loc),
    Named(name),
    Single<Expression>(single) {
  //
}

bi::UnaryCall::~UnaryCall() {
  //
}

bi::Expression* bi::UnaryCall::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::UnaryCall::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::UnaryCall::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
