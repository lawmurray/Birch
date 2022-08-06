/**
 * @file
 */
#include "src/expression/Super.hpp"

#include "src/visitor/all.hpp"

birch::Super::Super(Location* loc) :
    Expression(loc) {
  //
}

bool birch::Super::isSuper() const {
  return true;
}

void birch::Super::accept(Visitor* visitor) const {
  visitor->visit(this);
}

