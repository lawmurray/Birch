/**
 * @file
 */
#include "bi/statement/If.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::If::If(Expression* cond, Expression* braces,
    Expression* falseBraces, shared_ptr<Location> loc) :
    Statement(loc),
    Conditioned(cond),
    Braced(braces),
    falseBraces(falseBraces) {
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

bool bi::If::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::If::definitely(const If& o) const {
  return cond->definitely(*o.cond) && braces->definitely(*o.braces)
      && falseBraces->definitely(*o.falseBraces);
}

bool bi::If::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::If::possibly(const If& o) const {
  return cond->possibly(*o.cond) && braces->possibly(*o.braces)
      && falseBraces->possibly(*o.falseBraces);
}
