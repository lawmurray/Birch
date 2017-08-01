/**
 * @file
 */
#include "bi/statement/Return.hpp"

#include "bi/visitor/all.hpp"

bi::Return::Return(Expression* single,
    Location* loc) :
    Statement(loc),
    Unary<Expression>(single) {
  //
}

bi::Return::~Return() {
  //
}

bi::Statement* bi::Return::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Return::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Return::accept(Visitor* visitor) const {
  visitor->visit(this);
}
