/**
 * @file
 */
#include "bi/statement/Conditional.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Conditional::Conditional(Expression* cond, Expression* braces,
    Expression* falseBraces, shared_ptr<Location> loc) :
    Statement(loc),
    Conditioned(cond),
    Braced(braces),
    falseBraces(falseBraces) {
  /* pre-condition */
  assert(falseBraces);
}

bi::Conditional::~Conditional() {
  //
}

bi::Statement* bi::Conditional::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Conditional::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Conditional::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bi::possibly bi::Conditional::dispatch(Statement& o) {
  return o.le(*this);
}

bi::possibly bi::Conditional::le(Conditional& o) {
  return *cond <= *o.cond && *braces <= *o.braces
      && *falseBraces <= *o.falseBraces;
}
