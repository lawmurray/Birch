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

birch::Type* birch::TypeList::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::TypeList::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool birch::TypeList::isValue() const {
  for (auto type : *this) {
    if (!type->isValue()) {
      return false;
    }
  }
  return true;
}
