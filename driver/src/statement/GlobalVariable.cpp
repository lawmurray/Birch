/**
 * @file
 */
#include "src/statement/GlobalVariable.hpp"

#include "src/visitor/all.hpp"

birch::GlobalVariable::GlobalVariable(const Annotation annotation, Name* name,
    Type* type, Expression* brackets, Expression* args, Name* op,
    Expression* value, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    Typed(type),
    Bracketed(brackets),
    Argumented(args),
    Valued(op, value) {
  assert(value->isEmpty() || args->isEmpty());
}

void birch::GlobalVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
