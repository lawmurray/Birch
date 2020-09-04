/**
 * @file
 */
#include "src/statement/DoWhile.hpp"

#include "src/visitor/all.hpp"

birch::DoWhile::DoWhile(Statement* braces, Expression* cond, Location* loc) :
    Statement(loc),
    Scoped(LOCAL_SCOPE),
    Braced(braces),
    Conditioned(cond) {
  //
}

birch::DoWhile::~DoWhile() {
  //
}

birch::Statement* birch::DoWhile::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::DoWhile::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::DoWhile::accept(Visitor* visitor) const {
  visitor->visit(this);
}
