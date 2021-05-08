/**
 * @file
 */
#include "src/statement/If.hpp"

#include "src/visitor/all.hpp"

birch::If::If(Expression* cond, Statement* braces,
    Statement* falseBraces, Location* loc) :
    Statement(loc),
    Conditioned(cond),
    Braced(braces),
    falseBraces(falseBraces) {
  /* pre-condition */
  assert(falseBraces);
}

void birch::If::accept(Visitor* visitor) const {
  visitor->visit(this);
}
