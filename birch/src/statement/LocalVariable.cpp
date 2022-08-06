/**
 * @file
 */
#include "src/statement/LocalVariable.hpp"

#include "src/visitor/all.hpp"

birch::LocalVariable::LocalVariable(const Annotation annotation, Name* name,
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

birch::LocalVariable::LocalVariable(Expression* value, Location* loc) :
    LocalVariable(birch::LET, new Name(), new EmptyType(),
    new EmptyExpression(), new EmptyExpression(), new Name("<-"), value,
    loc) {
  //
}

birch::LocalVariable::LocalVariable(Name* name, Type* type, Location* loc) :
    LocalVariable(birch::LET, name, type, new EmptyExpression(),
    new EmptyExpression(), new Name(), new EmptyExpression(), loc) {
  //
}

void birch::LocalVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
