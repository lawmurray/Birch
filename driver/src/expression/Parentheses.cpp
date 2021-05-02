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

bool birch::Parentheses::isGlobal() const {
  return single->isGlobal();
}

bool birch::Parentheses::isMember() const {
  return single->isMember();
}

bool birch::Parentheses::isLocal() const {
  return single->isLocal();
}

bool birch::Parentheses::isParameter() const {
  return single->isParameter();
}

birch::Expression* birch::Parentheses::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Parentheses::accept(Visitor* visitor) const {
  visitor->visit(this);
}
