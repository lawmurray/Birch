/**
 * @file
 */
#include "bi/expression/Parameter.hpp"

#include "bi/visitor/all.hpp"

bi::Parameter::Parameter(Name* name, Type* type, Expression* value,
    Location* loc) :
    Expression(type, loc),
    Named(name),
    Valued(value) {
  //
}

bi::Parameter::~Parameter() {
  //
}

bi::Expression* bi::Parameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Parameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Parameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}
