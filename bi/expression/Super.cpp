/**
 * @file
 */
#include "bi/expression/Super.hpp"

#include "bi/visitor/all.hpp"

bi::Super::Super(Location* loc) :
    Expression(loc) {
  //
}

bi::Super::~Super() {
  //
}

bool bi::Super::isSuper() const {
  return true;
}

bi::Expression* bi::Super::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Super::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Super::accept(Visitor* visitor) const {
  visitor->visit(this);
}

