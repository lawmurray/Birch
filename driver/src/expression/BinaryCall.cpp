/**
 * @file
 */
#include "src/expression/BinaryCall.hpp"

#include "src/visitor/all.hpp"

birch::BinaryCall::BinaryCall(Expression* left, Name* name, Expression* right,
    Location* loc) :
    Expression(loc),
    Named(name),
    Couple<Expression>(left, right) {
  //
}

birch::BinaryCall::~BinaryCall() {
  //
}

birch::Expression* birch::BinaryCall::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Expression* birch::BinaryCall::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::BinaryCall::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
