/**
 * @file
 */
#include "bi/statement/While.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::While::While(Expression* cond, Expression* braces, shared_ptr<Location> loc) :
    Statement(loc),
    Conditioned(cond),
    Braced(braces) {
  //
}

bi::While::~While() {
  //
}

bi::Statement* bi::While::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::While::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::While::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::While::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::While::definitely(const While& o) const {
  return cond->definitely(*o.cond) && braces->definitely(*o.braces);
}

bool bi::While::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::While::possibly(const While& o) const {
  return cond->possibly(*o.cond) && braces->possibly(*o.braces);
}
