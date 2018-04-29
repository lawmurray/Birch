/**
 * @file
 */
#include "bi/expression/Assign.hpp"

#include "bi/visitor/all.hpp"

bi::Assign::Assign(Expression* left, Name* name, Expression* right,
    Location* loc, AssignmentOperator* target) :
    Expression(loc),
    Named(name),
    Couple<Expression>(left, right),
    Reference<AssignmentOperator>(target) {
  //
}

bi::Assign::~Assign() {
  //
}

bi::Expression* bi::Assign::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Assign::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Assign::accept(Visitor* visitor) const {
  visitor->visit(this);
}
