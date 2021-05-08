/**
 * @file
 */
#include "src/statement/With.hpp"

#include "src/visitor/all.hpp"

birch::With::With(Expression* single, Statement* braces, Location* loc) :
    Statement(loc),
    Single<Expression>(single),
    Braced(braces) {
  //
}

birch::Statement* birch::With::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::With::accept(Visitor* visitor) const {
  visitor->visit(this);
}
