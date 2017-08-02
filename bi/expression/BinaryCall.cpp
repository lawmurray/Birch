/**
 * @file
 */
#include "bi/expression/BinaryCall.hpp"

#include "bi/visitor/all.hpp"

bi::BinaryCall::BinaryCall(Expression* single, Expression* args,
    Location* loc) :
    Expression(loc),
    Single<Expression>(single),
    Argumented(args) {
  //
}

bi::BinaryCall::~BinaryCall() {
  //
}

bi::Expression* bi::BinaryCall::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::BinaryCall::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::BinaryCall::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
