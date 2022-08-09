/**
 * @file
 */
#include "src/expression/This.hpp"

#include "src/visitor/all.hpp"

birch::This::This(Location* loc) :
    Expression(loc) {
  //
}

bool birch::This::isThis() const {
  return true;
}

void birch::This::accept(Visitor* visitor) const {
  visitor->visit(this);
}
