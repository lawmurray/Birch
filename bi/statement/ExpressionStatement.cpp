/**
 * @file
 */
#include "bi/statement/ExpressionStatement.hpp"

#include "bi/visitor/all.hpp"

bi::ExpressionStatement::ExpressionStatement(Expression* single,
    Location* loc) :
    Statement(loc),
    Single<Expression>(single) {
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
