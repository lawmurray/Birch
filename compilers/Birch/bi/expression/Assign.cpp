/**
 * @file
 */
#include "bi/expression/Assign.hpp"

#include "bi/visitor/all.hpp"

bi::Assign::Assign(Expression* left, Name* op, Expression* right,
    Location* loc) :
    Expression(loc),
    Named(op),
    Couple<Expression>(left, right) {
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
