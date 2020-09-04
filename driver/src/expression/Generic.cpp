/**
 * @file
 */
#include "src/expression/Generic.hpp"

#include "src/visitor/all.hpp"

birch::Generic::Generic(const Annotation annotation, Name* name, Type* type,
    Location* loc) :
    Expression(loc),
    Annotated(annotation),
    Named(name),
    Typed(type) {
  //
}

birch::Generic::~Generic() {
  //
}

birch::Expression* birch::Generic::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Expression* birch::Generic::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Generic::accept(Visitor* visitor) const {
  visitor->visit(this);
}
