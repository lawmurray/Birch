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

bool bi::TypeList::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::TypeList::definitely(const TypeList& o) const {
  return head->definitely(*o.head) && tail->definitely(*o.tail);
}

bi::Type* bi::TypeList::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::TypeList::common(const TypeList& o) const {
  auto head1 = head->common(*o.head);
  auto tail1 = tail->common(*o.tail);
  if (head1 && tail1) {
    return new TypeList(head1, tail1);
  } else {
    return nullptr;
  }
}
