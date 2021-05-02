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

birch::Expression* birch::Super::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Super::accept(Visitor* visitor) const {
  visitor->visit(this);
}

