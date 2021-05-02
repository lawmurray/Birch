/**
 * @file
 */
#include "src/statement/If.hpp"

#include "src/visitor/all.hpp"

birch::If::If(Expression* cond, Statement* braces,
    Statement* falseBraces, Location* loc) :
    Statement(loc),
    Conditioned(cond),
    Scoped(LOCAL_SCOPE),
    Braced(braces),
    falseBraces(falseBraces),
    falseScope(new Scope(LOCAL_SCOPE)) {
  /* pre-condition */
  assert(falseBraces);
}

birch::Statement* birch::If::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::If::accept(Visitor* visitor) const {
  visitor->visit(this);
}
