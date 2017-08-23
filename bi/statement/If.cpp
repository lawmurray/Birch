/**
 * @file
 */
#include "bi/statement/If.hpp"

#include "bi/visitor/all.hpp"

bi::If::If(Expression* cond, Statement* braces,
    Statement* falseBraces, Location* loc) :
    Statement(loc),
    Conditioned(cond),
    Braced(braces),
    falseBraces(falseBraces),
    falseScope(new Scope()) {
  /* pre-condition */
  assert(falseBraces);
}

bi::If::~If() {
  //
}

bi::Statement* bi::If::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::If::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::If::accept(Visitor* visitor) const {
  visitor->visit(this);
}
