/**
 * @file
 */
#include "bi/expression/Nil.hpp"

#include "bi/visitor/all.hpp"

bi::Nil::Nil(Location* loc) :
    Expression(loc) {
  //
}

bi::Nil::~Nil() {
  //
}

bi::Expression* bi::Nil::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Nil::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Nil::accept(Visitor* visitor) const {
  visitor->visit(this);
}
