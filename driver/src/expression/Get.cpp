/**
 * @file
 */
#include "src/expression/Get.hpp"

#include "src/visitor/all.hpp"

birch::Get::Get(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

void birch::Get::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
