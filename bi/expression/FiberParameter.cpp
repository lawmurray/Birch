/**
 * @file
 */
#include "bi/expression/FiberParameter.hpp"

#include "bi/visitor/all.hpp"

bi::FiberParameter::FiberParameter(const Annotation annotation, Name* name,
    Type* type, Expression* value, Location* loc) :
    Expression(type, loc),
    Annotated(annotation),
    Named(name),
    Valued(value) {
  //
}

bi::FiberParameter::~FiberParameter() {
  //
}

bi::Expression* bi::FiberParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::FiberParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::FiberParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}
