/**
 * @file
 */
#include "bi/type/EmptyType.hpp"

#include "bi/visitor/all.hpp"

bi::EmptyType::EmptyType(const bool assignable) :
    Type(nullptr, assignable) {
  //
}

bi::EmptyType::~EmptyType() {
  //
}

bi::Type* bi::EmptyType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::EmptyType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::EmptyType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::EmptyType::isEmpty() const {
  return true;
}

bool bi::EmptyType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::EmptyType::definitely(const EmptyType& o) const {
  return true;
}

bool bi::EmptyType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::EmptyType::possibly(const EmptyType& o) const {
  return true;
}
