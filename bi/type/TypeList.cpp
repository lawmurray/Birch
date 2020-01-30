/**
 * @file
 */
#include "bi/type/TypeList.hpp"

#include "bi/visitor/all.hpp"

bi::TypeList::TypeList(Type* head, Type* tail, Location* loc) :
    Type(loc),
    head(head),
    tail(tail) {
  /* pre-conditions */
  assert(head);
  assert(tail);
}

bi::TypeList::~TypeList() {
  //
}

bool bi::TypeList::isList() const {
  return true;
}

bi::Type* bi::TypeList::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::TypeList::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::TypeList::accept(Visitor* visitor) const {
  visitor->visit(this);
}
