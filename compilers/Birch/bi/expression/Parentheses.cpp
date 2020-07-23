/**
 * @file
 */
#include "bi/expression/Parentheses.hpp"

#include "bi/visitor/all.hpp"

bi::Parentheses::Parentheses(Expression* single,
    Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

bi::Parentheses::~Parentheses() {
  //
}

const bi::Expression* bi::Parentheses::strip() const {
  return single->strip();
}

bool bi::Parentheses::isAssignable() const {
  return single->isAssignable();
}

bool bi::Parentheses::isSlice() const {
  return single->isSlice();
}

bool bi::Parentheses::isTuple() const {
  return single->isTuple();
}

bool bi::Parentheses::isMembership() const {
  return single->isMembership();
}

bool bi::Parentheses::isGlobal() const {
  return single->isGlobal();
}

bool bi::Parentheses::isMember() const {
  return single->isMember();
}

bool bi::Parentheses::isLocal() const {
  return single->isLocal();
}

bool bi::Parentheses::isParameter() const {
  return single->isParameter();
}

bi::Expression* bi::Parentheses::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Parentheses::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Parentheses::accept(Visitor* visitor) const {
  visitor->visit(this);
}
