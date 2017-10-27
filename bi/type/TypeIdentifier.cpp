/**
 * @file
 */
#include "bi/type/TypeIdentifier.hpp"

#include "bi/visitor/all.hpp"

bi::TypeIdentifier::TypeIdentifier(Name* name,
    Location* loc, const bool assignable) :
    Type(loc, assignable),
    Named(name) {
  //
}

bi::TypeIdentifier::~TypeIdentifier() {
  //
}

bi::Type* bi::TypeIdentifier::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::TypeIdentifier::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::TypeIdentifier::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::TypeIdentifier::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}
