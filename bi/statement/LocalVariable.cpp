/**
 * @file
 */
#include "bi/statement/LocalVariable.hpp"

#include "bi/visitor/all.hpp"

bi::LocalVariable::LocalVariable(const Annotation annotation, Name* name,
    Type* type, Expression* brackets, Expression* args, Expression* value,
    Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    Typed(type),
    Bracketed(brackets),
    Argumented(args),
    Valued(value) {
  assert(value->isEmpty() || args->isEmpty());
}

bi::LocalVariable::LocalVariable(Expression* value, Location* loc) :
    Statement(loc),
    Annotated(bi::AUTO),
    Named(new Name()),
    Typed(new EmptyType()),
    Bracketed(new EmptyExpression()),
    Argumented(new EmptyExpression()),
    Valued(value) {
  //
}

bi::LocalVariable::~LocalVariable() {
  //
}

bool bi::LocalVariable::needsConstruction() const {
  return !args->isEmpty()
      || (value->isEmpty() && (!type->isArray() || !brackets->isEmpty()));
}

bi::Statement* bi::LocalVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::LocalVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::LocalVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
