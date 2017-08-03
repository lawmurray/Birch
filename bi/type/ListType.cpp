/**
 * @file
 */
#include "bi/common/List.hpp"

#include "bi/visitor/all.hpp"

bi::ListType::ListType(Type* head, Type* tail, Location* loc) :
    head(head),
    tail(tail) {
  /* pre-conditions */
  assert(head);
  assert(tail);

  this->loc = loc;
}

bi::ListType::~ListType() {
  //
}

bi::Type* bi::ListType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::ListType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ListType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ListType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::ListType::definitely(const AliasType& o) const {
  return definitely(*o.target->base);
}

bool bi::ListType::definitely(const ListType& o) const {
  return head->definitely(*o.head) && tail->definitely(*o.tail);
}

bool bi::ListType::definitely(const OptionalType& o) const {
  return definitely(*o.single);
}

bool bi::ListType::definitely(const ParenthesesType& o) const {
  return definitely(*o.single);
}

bool bi::ListType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::ListType::possibly(const AliasType& o) const {
  return possibly(*o.target->base);
}

bool bi::ListType::possibly(const ListType& o) const {
  return head->possibly(*o.head) && tail->possibly(*o.tail);
}

bool bi::ListType::possibly(const OptionalType& o) const {
  return possibly(*o.single);
}

bool bi::ListType::possibly(const ParenthesesType& o) const {
  return possibly(*o.single);
}
