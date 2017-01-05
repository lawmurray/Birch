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

bool bi::ExpressionStatement::dispatch(Statement& o) {
  return o.le(*this);
}

bool bi::ExpressionStatement::le(ExpressionStatement& o) {
  return *single <= *o.single;
}
