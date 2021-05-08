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

void birch::Return::accept(Visitor* visitor) const {
  visitor->visit(this);
}
