/**
 * @file
 */
#include "bi/expression/ExpressionList.hpp"

#include "bi/visitor/all.hpp"

bi::ExpressionList::ExpressionList(Expression* head, Expression* tail,
    Location* loc) :
    Expression(loc),
    head(head),
    tail(tail) {
  /* pre-conditions */
  assert(head);
  assert(tail);

  this->loc = loc;
}

bi::ExpressionList::~ExpressionList() {
  //
}

bool bi::ExpressionList::isAssignable() const {
  return head->isAssignable() && tail->isAssignable();
}

bool bi::ExpressionList::isList() const {
  return true;
}

bi::Expression* bi::ExpressionList::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::ExpressionList::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ExpressionList::accept(Visitor* visitor) const {
  visitor->visit(this);
}
