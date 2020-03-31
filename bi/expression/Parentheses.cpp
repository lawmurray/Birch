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

bi::Expression* bi::Parentheses::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Parentheses::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Parentheses::accept(Visitor* visitor) const {
  visitor->visit(this);
}
