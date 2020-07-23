/**
 * @file
 */
#include "bi/statement/Braces.hpp"

#include "bi/visitor/all.hpp"

bi::Braces::Braces(Statement* single, Location* loc) :
    Statement(loc),
    Single<Statement>(single) {
  //
}

bi::Braces::~Braces() {
  //
}

bi::Statement* bi::Braces::strip() {
  return single->strip();
}

bi::Statement* bi::Braces::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Braces::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Braces::accept(Visitor* visitor) const {
  visitor->visit(this);
}
