/**
 * @file
 */
#include "bi/common/List.hpp"

#include "bi/visitor/all.hpp"

bi::TypeList::TypeList(Type* head, Type* tail, Location* loc,
    const bool assignable) :
    Type(loc, assignable),
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

bool bi::TypeList::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::TypeList::definitely(const AliasType& o) const {
  assert(o.target);
  return definitely(*o.target->base);
}

bool bi::TypeList::definitely(const TypeList& o) const {
  return head->definitely(*o.head) && tail->definitely(*o.tail);
}

bool bi::TypeList::definitely(const OptionalType& o) const {
  return definitely(*o.single);
}

bool bi::TypeList::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::TypeList::possibly(const AliasType& o) const {
  assert(o.target);
  return possibly(*o.target->base);
}

bool bi::TypeList::possibly(const TypeList& o) const {
  return head->possibly(*o.head) && tail->possibly(*o.tail);
}

bool bi::TypeList::possibly(const OptionalType& o) const {
  return possibly(*o.single);
}
