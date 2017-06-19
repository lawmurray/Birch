/**
 * @file
 */
#include "bi/type/IdentifierType.hpp"

#include "bi/visitor/all.hpp"

bi::IdentifierType::IdentifierType(shared_ptr<Name> name,
    shared_ptr<Location> loc) :
    Type(loc),
    Named(name) {
  //
}

bi::IdentifierType::~IdentifierType() {
  //
}

bi::Type* bi::IdentifierType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::IdentifierType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::IdentifierType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::IdentifierType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::IdentifierType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}
