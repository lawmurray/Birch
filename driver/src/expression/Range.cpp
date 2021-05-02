/**
 * @file
 */
#include "src/expression/Range.hpp"

#include "src/visitor/all.hpp"

birch::Range::Range(Expression* left, Expression* right,
    Location* loc) :
    Expression(loc),
    Couple<Expression>(left, right) {
  //
}

birch::Expression* birch::Range::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Range::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
