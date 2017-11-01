/**
 * @file
 */
#include "bi/type/AnyType.hpp"

#include "bi/visitor/all.hpp"

bi::AnyType::AnyType(Location* loc) :
    Type(loc) {
  //
}

bi::AnyType::~AnyType() {
  //
}

bi::Type* bi::AnyType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::AnyType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::AnyType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::AnyType::isEmpty() const {
  return true;
}

bool bi::AnyType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::AnyType::definitely(const AnyType& o) const {
  return true;
}

bool bi::AnyType::definitely(const GenericType& o) const {
  assert(o.target);
  return definitely(*o.target->type);
}
