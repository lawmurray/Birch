/**
 * @file
 */
#include "src/statement/Yield.hpp"

#include "src/visitor/all.hpp"

birch::Yield::Yield(Expression* single,
    Location* loc) :
    Statement(loc),
    Single<Expression>(single),
    resume(nullptr) {
  //
}

birch::Yield::~Yield() {
  //
}

birch::Statement* birch::Yield::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::Yield::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Yield::accept(Visitor* visitor) const {
  visitor->visit(this);
}
