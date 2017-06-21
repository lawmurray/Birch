/**
 * @file
 */
#include "bi/statement/Assert.hpp"

#include "bi/visitor/all.hpp"

bi::Assert::Assert(Expression* cond, shared_ptr<Location> loc) :
    Statement(loc),
    Conditioned(cond) {
  //
}

bi::Assert::~Assert() {
  //
}

bi::Statement* bi::Assert::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Assert::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Assert::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Assert::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::Assert::definitely(const Assert& o) const {
  return cond->definitely(*o.cond);
}

bool bi::Assert::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::Assert::possibly(const Assert& o) const {
  return cond->possibly(*o.cond);
}
