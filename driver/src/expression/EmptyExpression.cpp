/**
 * @file
 */
#include "src/expression/EmptyExpression.hpp"

#include "src/visitor/all.hpp"

birch::EmptyExpression::EmptyExpression(Location* loc) :
    Expression(loc) {
  //
}

birch::Expression* birch::EmptyExpression::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::EmptyExpression::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool birch::EmptyExpression::isEmpty() const {
  return true;
}
