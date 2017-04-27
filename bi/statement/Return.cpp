/**
 * @file
 */
#include "bi/statement/Return.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Return::Return(Expression* single,
    shared_ptr<Location> loc) :
    Statement(loc),
    ExpressionUnary(single) {
  //
}

bi::Return::~Return() {
  //
}

bi::Statement* bi::Return::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Return::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Return::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Return::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::Return::definitely(const Return& o) const {
  return single->definitely(*o.single);
}

bool bi::Return::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::Return::possibly(const Return& o) const {
  return single->possibly(*o.single);
}
