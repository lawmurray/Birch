/**
 * @file
 */
#include "bi/expression/Global.hpp"

#include "bi/visitor/all.hpp"

bi::Global::Global(Location* loc) :
    Expression(loc) {
  //
}

bi::Global::~Global() {
  //
}

bi::Expression* bi::Global::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Global::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Global::accept(Visitor* visitor) const {
  visitor->visit(this);
}
