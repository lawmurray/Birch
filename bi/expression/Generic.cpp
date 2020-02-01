/**
 * @file
 */
#include "bi/expression/Generic.hpp"

#include "bi/visitor/all.hpp"

bi::Generic::Generic(const Annotation annotation, Name* name, Type* type,
    Location* loc) :
    Expression(loc),
    Annotated(annotation),
    Named(name),
    Typed(type) {
  //
}

bi::Generic::~Generic() {
  //
}

bi::Expression* bi::Generic::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Generic::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Generic::accept(Visitor* visitor) const {
  visitor->visit(this);
}
