/**
 * @file
 */
#include "src/statement/ExpressionStatement.hpp"

#include "src/visitor/all.hpp"

birch::ExpressionStatement::ExpressionStatement(Expression* single,
    Location* loc) :
    Statement(loc),
    Single<Expression>(single) {
  //
}

birch::ExpressionStatement::~ExpressionStatement() {
  //
}

birch::Statement* birch::ExpressionStatement::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::ExpressionStatement::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::ExpressionStatement::accept(Visitor* visitor) const {
  visitor->visit(this);
}
