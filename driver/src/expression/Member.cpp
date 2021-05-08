/**
 * @file
 */
#include "src/expression/Member.hpp"

#include "src/visitor/all.hpp"

birch::Member::Member(Expression* left, Expression* right,
    Location* loc) :
    Expression(loc),
    Couple<Expression>(left, right) {
  //
}

bool birch::Member::isMembership() const {
  return true;
}

void birch::Member::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
