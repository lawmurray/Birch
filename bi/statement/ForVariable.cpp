/**
 * @file
 */
#include "bi/statement/ForVariable.hpp"

#include "bi/visitor/all.hpp"

bi::ForVariable::ForVariable(const Annotation annotation, Name* name,
    Type* type, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    Typed(type) {
  //
}

bi::ForVariable::~ForVariable() {
  //
}

bi::Statement* bi::ForVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::ForVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ForVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
