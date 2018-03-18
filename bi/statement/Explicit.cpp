/**
 * @file
 */
#include "bi/statement/Explicit.hpp"

#include "bi/visitor/all.hpp"

bi::Explicit::Explicit(Type* base, Location* loc) :
    Statement(loc),
    Based(base, false) {
  //
}

bi::Explicit::~Explicit() {
  //
}

bi::Statement* bi::Explicit::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Explicit::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Explicit::accept(Visitor* visitor) const {
  visitor->visit(this);
}
