/**
 * @file
 */
#include "src/expression/ExpressionList.hpp"

#include "src/visitor/all.hpp"

birch::ExpressionList::ExpressionList(Expression* head, Expression* tail,
    Location* loc) :
    Expression(loc),
    head(head),
    tail(tail) {
  /* pre-conditions */
  assert(head);
  assert(tail);

  this->loc = loc;
}

bool birch::ExpressionList::isTuple() const {
  return true;
}

void birch::ExpressionList::accept(Visitor* visitor) const {
  visitor->visit(this);
}
