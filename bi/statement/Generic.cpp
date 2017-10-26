/**
 * @file
 */
#include "bi/statement/Generic.hpp"

#include "bi/visitor/all.hpp"

bi::Generic::Generic(Name* name, Location* loc) :
    Statement(loc),
    Named(name) {
  //
}

bi::Generic::~Generic() {
  //
}

bi::Statement* bi::Generic::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Generic::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Generic::accept(Visitor* visitor) const {
  visitor->visit(this);
}
