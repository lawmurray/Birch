/**
 * @file
 */
#include "src/statement/Return.hpp"

#include "src/visitor/all.hpp"

birch::Return::Return(Expression* single,
    Location* loc) :
    Statement(loc),
    Single<Expression>(single) {
  //
}

birch::Return::~Return() {
  //
}

birch::Statement* birch::Return::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::Return::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Return::accept(Visitor* visitor) const {
  visitor->visit(this);
}
