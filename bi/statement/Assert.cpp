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
