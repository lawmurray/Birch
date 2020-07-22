/**
 * @file
 */
#include "bi/statement/DoWhile.hpp"

#include "bi/visitor/all.hpp"

bi::DoWhile::DoWhile(Statement* braces, Expression* cond, Location* loc) :
    Statement(loc),
    Scoped(LOCAL_SCOPE),
    Braced(braces),
    Conditioned(cond) {
  //
}

bi::DoWhile::~DoWhile() {
  //
}

bi::Statement* bi::DoWhile::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::DoWhile::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::DoWhile::accept(Visitor* visitor) const {
  visitor->visit(this);
}
