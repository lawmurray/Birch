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

void birch::Generic::accept(Visitor* visitor) const {
  visitor->visit(this);
}
