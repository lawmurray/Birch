/**
 * @file
 */
#include "src/expression/Global.hpp"

#include "src/visitor/all.hpp"

birch::Global::Global(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

birch::Global::~Global() {
  //
}

birch::Expression* birch::Global::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Expression* birch::Global::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Global::accept(Visitor* visitor) const {
  visitor->visit(this);
}
