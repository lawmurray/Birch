/**
 * @file
 */
#include "bi/expression/Member.hpp"

#include "bi/visitor/all.hpp"

bi::Member::Member(Expression* left, Expression* right,
    Location* loc) :
    Expression(loc),
    Couple<Expression>(left, right) {
  //
}

bi::Member::~Member() {
  //
}

bool bi::Member::isAssignable() const {
  return right->isAssignable();
}

bool bi::Member::isMember() const {
  return true;
}

bi::Expression* bi::Member::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Member::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Member::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
