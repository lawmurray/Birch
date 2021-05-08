/**
 * @file
 */
#include "src/type/TypeList.hpp"

#include "src/visitor/all.hpp"

birch::TypeList::TypeList(Type* head, Type* tail, Location* loc) :
    Type(loc),
    head(head),
    tail(tail) {
  /* pre-conditions */
  assert(head);
  assert(tail);
}

void birch::TypeList::accept(Visitor* visitor) const {
  visitor->visit(this);
}
