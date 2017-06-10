/**
 * @file
 */
#include "bi/type/CoroutineType.hpp"

#include "bi/visitor/all.hpp"

bi::CoroutineType::CoroutineType(Type* type,
    shared_ptr<Location> loc, const bool assignable) :
    Type(loc, assignable),
    type(type) {
  //
}

bi::CoroutineType::~CoroutineType() {
  //
}

bi::Type* bi::CoroutineType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::CoroutineType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::CoroutineType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::CoroutineType::isCoroutine() const {
  return true;
}

bool bi::CoroutineType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::CoroutineType::definitely(const CoroutineType& o) const {
  return type->definitely(*o.type);
}

bool bi::CoroutineType::definitely(const ParenthesesType& o) const {
  return definitely(*o.single);
}

bool bi::CoroutineType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::CoroutineType::possibly(const CoroutineType& o) const {
  return type->possibly(*o.type);
}

bool bi::CoroutineType::possibly(const ParenthesesType& o) const {
  return possibly(*o.single);
}
