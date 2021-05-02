/**
 * @file
 */
#include "src/expression/Parameter.hpp"

#include "src/visitor/all.hpp"

birch::Parameter::Parameter(const Annotation annotation, Name* name,
    Type* type, Name* op, Expression* value, Location* loc) :
    Expression(loc),
    Annotated(annotation),
    Named(name),
    Typed(type),
    Valued(op, value) {
  //
}

birch::Expression* birch::Parameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Parameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}
