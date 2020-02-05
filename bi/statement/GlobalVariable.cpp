/**
 * @file
 */
#include "bi/statement/GlobalVariable.hpp"

#include "bi/visitor/all.hpp"

bi::GlobalVariable::GlobalVariable(const Annotation annotation, Name* name, Type* type,
    Expression* brackets, Expression* args, Expression* value, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    Typed(type),
    Bracketed(brackets),
    Argumented(args),
    Valued(value) {
  assert(value->isEmpty() || args->isEmpty());
}

bi::GlobalVariable::~GlobalVariable() {
  //
}

bi::Statement* bi::GlobalVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::GlobalVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::GlobalVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
