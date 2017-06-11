/**
 * @file
 */
#include "bi/statement/ExpressionStatement.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ExpressionStatement::ExpressionStatement(Expression* single,
    shared_ptr<Location> loc) :
    Statement(loc),
    Unary<Expression>(single) {
  //
}

bi::ExpressionStatement::~ExpressionStatement() {
  //
}

bi::Statement* bi::ExpressionStatement::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::ExpressionStatement::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ExpressionStatement::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ExpressionStatement::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::ExpressionStatement::definitely(const ExpressionStatement& o) const {
  return single->definitely(*o.single);
}

bool bi::ExpressionStatement::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::ExpressionStatement::possibly(const ExpressionStatement& o) const {
  return single->possibly(*o.single);
}
