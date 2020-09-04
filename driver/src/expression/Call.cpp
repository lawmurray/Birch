/**
 * @file
 */
#include "src/expression/Call.hpp"

#include "src/visitor/all.hpp"

birch::Call::Call(Expression* single, Expression* args, Location* loc) :
    Expression(loc),
    Single<Expression>(single),
    Argumented(args) {
  //
}

birch::Call::Call(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single),
    Argumented(new EmptyExpression()) {
  //
}

birch::Call::~Call() {
  //
}

birch::Expression* birch::Call::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Expression* birch::Call::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Call::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
