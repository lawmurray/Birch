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

int bi::ExpressionList::count() const {
  const ExpressionList* listTail = dynamic_cast<const ExpressionList*>(tail);
  if (listTail) {
    return 1 + listTail->count();
  } else {
    return 2;
  }
}

int bi::ExpressionList::rangeCount() const {
  const Range* rangeHead = dynamic_cast<const Range*>(head);
  const Range* rangeTail = dynamic_cast<const Range*>(tail);
  const ExpressionList* listTail = dynamic_cast<const ExpressionList*>(tail);
  int count = 0;

  if (rangeHead) {
    ++count;
  }
  if (rangeTail) {
    ++count;
  } else if (listTail) {
    count += listTail->rangeCount();
  }
  return count;
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
