/**
 * @file
 */
#include "bi/expression/BracesExpression.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::BracesExpression::BracesExpression(Statement* single,
    shared_ptr<Location> loc) :
    Expression(loc),
    StatementUnary(single) {
  //
}

bi::BracesExpression::~BracesExpression() {
  //
}

bi::Expression* bi::BracesExpression::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::BracesExpression::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::BracesExpression::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::BracesExpression::dispatchDefinitely(Expression& o) {
  return o.definitely(*this);
}

bool bi::BracesExpression::definitely(BracesExpression& o) {
  return single->definitely(*o.single) && type->definitely(*o.type);
}

bool bi::BracesExpression::dispatchPossibly(Expression& o) {
  return o.possibly(*this);
}

bool bi::BracesExpression::possibly(BracesExpression& o) {
  return single->possibly(*o.single) && type->possibly(*o.type);
}
