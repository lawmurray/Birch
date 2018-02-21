/**
 * @file
 */
#include "bi/statement/MemberVariable.hpp"

#include "bi/visitor/all.hpp"

bi::MemberVariable::MemberVariable(const Annotation annotation, Name* name,
    Type* type, Expression* brackets, Expression* args, Expression* value,
    Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    Typed(type),
    Bracketed(brackets),
    Argumented(args),
    Valued(value) {
  //
}

bi::MemberVariable::~MemberVariable() {
  //
}

bool bi::MemberVariable::needsConstruction() const {
  return !args->isEmpty()
      || (value->isEmpty() && (!type->isArray() || !brackets->isEmpty()));
}

bi::Statement* bi::MemberVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::MemberVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::MemberVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
