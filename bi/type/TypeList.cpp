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

bool bi::TypeList::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bool bi::TypeList::isConvertible(const TypeList& o) const {
  return head->isConvertible(*o.head) && tail->isConvertible(*o.tail);
}

bool bi::TypeList::dispatchIsAssignable(const Type& o) const {
  return o.isAssignable(*this);
}

bool bi::TypeList::isAssignable(const TypeList& o) const {
  return head->isAssignable(*o.head) && tail->isAssignable(*o.tail);
}
