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

birch::ExpressionList::~ExpressionList() {
  //
}

bool birch::ExpressionList::isAssignable() const {
  return head->isAssignable() && tail->isAssignable();
}

bool birch::ExpressionList::isTuple() const {
  return true;
}

birch::Expression* birch::ExpressionList::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Expression* birch::ExpressionList::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::ExpressionList::accept(Visitor* visitor) const {
  visitor->visit(this);
}
