/**
 * @file
 */
#include "src/statement/Assert.hpp"

#include "src/visitor/all.hpp"

birch::Assert::Assert(Expression* cond, Location* loc) :
    Statement(loc),
    Conditioned(cond) {
  //
}

birch::Assert::~Assert() {
  //
}

birch::Statement* birch::Assert::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::Assert::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Assert::accept(Visitor* visitor) const {
  visitor->visit(this);
}
