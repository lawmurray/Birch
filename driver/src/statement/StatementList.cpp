/**
 * @file
 */
#include "src/statement/StatementList.hpp"

#include "src/visitor/all.hpp"

birch::StatementList::StatementList(Statement* head, Statement* tail,
    Location* loc) :
    Statement(loc),
    head(head),
    tail(tail) {
  /* pre-conditions */
  assert(head);
  assert(tail);

  this->loc = loc;
}

int birch::StatementList::count() const {
  const StatementList* listTail = dynamic_cast<const StatementList*>(tail);
  if (listTail) {
    return 1 + listTail->count();
  } else {
    return 2;
  }
}

bool birch::StatementList::isEmpty() const {
  return head->isEmpty() && tail->isEmpty();
}

void birch::StatementList::accept(Visitor* visitor) const {
  visitor->visit(this);
}
