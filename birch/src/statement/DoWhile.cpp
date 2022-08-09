/**
 * @file
 */
#include "src/statement/DoWhile.hpp"

#include "src/visitor/all.hpp"

birch::DoWhile::DoWhile(Statement* braces, Expression* cond, Location* loc) :
    Statement(loc),
    Braced(braces),
    Conditioned(cond) {
  //
}

void birch::DoWhile::accept(Visitor* visitor) const {
  visitor->visit(this);
}
