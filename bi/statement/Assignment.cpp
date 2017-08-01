/**
 * @file
 */
#include "bi/statement/Assignment.hpp"

#include "bi/visitor/all.hpp"

bi::Assignment::Assignment(Expression* left, Expression* right,
    Location* loc, AssignmentOperator* target) :
    Statement(loc),
    Couple<Expression>(left, right),
    Reference<AssignmentOperator>(target) {
  //
}

bi::Assignment::~Assignment() {
  //
}

bi::Statement* bi::Assignment::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Assignment::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Assignment::accept(Visitor* visitor) const {
  visitor->visit(this);
}
