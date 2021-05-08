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

void birch::Assert::accept(Visitor* visitor) const {
  visitor->visit(this);
}
