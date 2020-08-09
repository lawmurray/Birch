/**
 * @file
 */
#include "bi/statement/Assume.hpp"

#include "bi/visitor/all.hpp"

bi::Assume::Assume(Expression* left, Name* op, Expression* right,
    Location* loc) :
    Statement(loc),
    Named(op),
    Couple<Expression>(left, right) {
  //
}

bi::Assume::~Assume() {
  //
}

bi::Statement* bi::Assume::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Assume::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Assume::accept(Visitor* visitor) const {
  visitor->visit(this);
}
