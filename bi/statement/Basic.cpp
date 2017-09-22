/**
 * @file
 */
#include "bi/statement/Basic.hpp"

#include "bi/visitor/all.hpp"

bi::Basic::Basic(Name* name, Type* base, Location* loc) :
    Statement(loc),
    Named(name),
    Based(base) {
  //
}

bi::Basic::~Basic() {
  //
}

bi::Statement* bi::Basic::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Basic::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Basic::accept(Visitor* visitor) const {
  visitor->visit(this);
}
