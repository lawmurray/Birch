/**
 * @file
 */
#include "bi/expression/Local.hpp"

#include "bi/visitor/all.hpp"

bi::Local::Local(Location* loc) :
    Expression(loc) {
  //
}

bi::Local::~Local() {
  //
}

bi::Expression* bi::Local::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Local::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Local::accept(Visitor* visitor) const {
  visitor->visit(this);
}
