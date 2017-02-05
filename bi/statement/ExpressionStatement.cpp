/**
 * @file
 */
#include "bi/statement/ExpressionStatement.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ExpressionStatement::ExpressionStatement(Expression* single,
    shared_ptr<Location> loc) :
    Statement(loc),
    ExpressionUnary(single) {
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

bool bi::ExpressionStatement::dispatchDefinitely(Statement& o) {
  return o.definitely(*this);
}

bool bi::ExpressionStatement::definitely(ExpressionStatement& o) {
  return single->definitely(*o.single);
}

bool bi::ExpressionStatement::dispatchPossibly(Statement& o) {
  return o.possibly(*this);
}

bool bi::ExpressionStatement::possibly(ExpressionStatement& o) {
  return single->possibly(*o.single);
}
