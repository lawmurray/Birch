/**
 * @file
 */
#include "src/statement/Factor.hpp"

#include "src/visitor/all.hpp"

birch::Factor::Factor(Expression* single,
    Location* loc) :
    Statement(loc),
    Single<Expression>(single) {
  //
}

birch::Statement* birch::Factor::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Factor::accept(Visitor* visitor) const {
  visitor->visit(this);
}
