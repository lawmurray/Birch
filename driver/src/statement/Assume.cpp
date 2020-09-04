/**
 * @file
 */
#include "src/statement/Assume.hpp"

#include "src/visitor/all.hpp"

birch::Assume::Assume(Expression* left, Name* op, Expression* right,
    Location* loc) :
    Statement(loc),
    Named(op),
    Couple<Expression>(left, right) {
  //
}

birch::Assume::~Assume() {
  //
}

birch::Statement* birch::Assume::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::Assume::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Assume::accept(Visitor* visitor) const {
  visitor->visit(this);
}
