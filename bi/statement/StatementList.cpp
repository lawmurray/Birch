/**
 * @file
 */
#include "bi/statement/StatementList.hpp"

#include "bi/visitor/all.hpp"

bi::StatementList::StatementList(Statement* head, Statement* tail,
    Location* loc) :
    Statement(loc),
    head(head),
    tail(tail) {
  /* pre-conditions */
  assert(head);
  assert(tail);

  this->loc = loc;
}

bi::StatementList::~StatementList() {
  //
}

int bi::StatementList::count() const {
  const StatementList* listTail = dynamic_cast<const StatementList*>(tail);
  if (listTail) {
    return 1 + listTail->count();
  } else {
    return 2;
  }
}

bi::Statement* bi::StatementList::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::StatementList::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::StatementList::accept(Visitor* visitor) const {
  visitor->visit(this);
}
