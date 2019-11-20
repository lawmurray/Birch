/**
 * @file
 */
#include "bi/statement/FiberVariable.hpp"

#include "bi/visitor/all.hpp"

bi::FiberVariable::FiberVariable(const Annotation annotation, Name* name,
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

bi::FiberVariable::FiberVariable(Expression* value, Location* loc) :
    Statement(loc),
    Annotated(bi::AUTO),
    Named(new Name()),
    Typed(new EmptyType()),
    Bracketed(new EmptyExpression()),
    Argumented(new EmptyExpression()),
    Valued(value) {
  //
}

bi::FiberVariable::~FiberVariable() {
  //
}

bool bi::FiberVariable::needsConstruction() const {
  return !args->isEmpty()
      || (value->isEmpty() && (!type->isArray() || !brackets->isEmpty()));
}

bi::Statement* bi::FiberVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::FiberVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::FiberVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
