/**
 * @file
 */
#include "bi/expression/Super.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Super::Super(shared_ptr<Location> loc) :
    Expression(loc) {
  //
}

bi::Super::~Super() {
  //
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
