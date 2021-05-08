/**
 * @file
 */
#include "src/statement/Braces.hpp"

#include "src/visitor/all.hpp"

birch::Braces::Braces(Statement* single, Location* loc) :
    Statement(loc),
    Single<Statement>(single) {
  //
}

birch::Statement* birch::Braces::strip() {
  return single->strip();
}

void birch::Braces::accept(Visitor* visitor) const {
  visitor->visit(this);
}
