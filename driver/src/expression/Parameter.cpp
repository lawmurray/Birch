/**
 * @file
 */
#include "src/expression/Parameter.hpp"

#include "src/visitor/all.hpp"

birch::Parameter::Parameter(const Annotation annotation, Name* name, Type* type,
    Expression* value, Location* loc) :
    Expression(loc),
    Annotated(annotation),
    Named(name),
    Typed(type),
    Valued(value) {
  //
}

birch::Parameter::~Parameter() {
  //
}

birch::Expression* birch::Parameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Expression* birch::Parameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Parameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}
