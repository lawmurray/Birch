/**
 * @file
 */
#include "src/expression/Parentheses.hpp"

#include "src/visitor/all.hpp"

birch::Parentheses::Parentheses(Expression* single,
    Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

const birch::Expression* birch::Parentheses::strip() const {
  return single->strip();
}

bool birch::Parentheses::isSlice() const {
  return single->isSlice();
}

bool birch::Parentheses::isTuple() const {
  return single->isTuple();
}

bool birch::Parentheses::isMembership() const {
  return single->isMembership();
}

void birch::Parentheses::accept(Visitor* visitor) const {
  visitor->visit(this);
}
