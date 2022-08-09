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

void birch::ExpressionStatement::accept(Visitor* visitor) const {
  visitor->visit(this);
}
