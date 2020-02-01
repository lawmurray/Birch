/**
 * @file
 */
#include "bi/expression/Parameter.hpp"

#include "bi/visitor/all.hpp"

bi::Parameter::Parameter(const Annotation annotation, Name* name, Type* type,
    Expression* value, Location* loc) :
    Expression(loc),
    Annotated(annotation),
    Named(name),
    Typed(type),
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
