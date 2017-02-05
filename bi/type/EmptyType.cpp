/**
 * @file
 */
#include "bi/type/EmptyType.hpp"

#include "bi/visitor/all.hpp"

bi::EmptyType::EmptyType(shared_ptr<Location> loc) :
    Type(loc) {
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

bool bi::EmptyType::dispatchDefinitely(Type& o) {
  return o.definitely(*this);
}

bool bi::EmptyType::definitely(EmptyType& o) {
  return !o.assignable || assignable;
}

bool bi::EmptyType::dispatchPossibly(Type& o) {
  return o.possibly(*this);
}

bool bi::EmptyType::possibly(EmptyType& o) {
  return !o.assignable || assignable;
}
