/**
 * @file
 */
#include "bi/statement/ParallelVariable.hpp"

#include "bi/visitor/all.hpp"

bi::ParallelVariable::ParallelVariable(const Annotation annotation, Name* name,
    Type* type, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    Typed(type) {
  //
}

bi::ParallelVariable::~ParallelVariable() {
  //
}

bi::Statement* bi::ParallelVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::ParallelVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ParallelVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
