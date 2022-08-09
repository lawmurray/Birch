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

void birch::Global::accept(Visitor* visitor) const {
  visitor->visit(this);
}
