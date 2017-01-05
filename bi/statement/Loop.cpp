/**
 * @file
 */
#include "bi/statement/Loop.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Loop::Loop(Expression* cond, Expression* braces, shared_ptr<Location> loc) :
    Statement(loc),
    Conditioned(cond),
    Braced(braces) {
  //
}

bi::Loop::~Loop() {
  //
}

bi::Statement* bi::Loop::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Loop::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Loop::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Loop::dispatch(Statement& o) {
  return o.le(*this);
}

bool bi::Loop::le(Loop& o) {
  return *cond <= *o.cond && *braces <= *o.braces;
}
