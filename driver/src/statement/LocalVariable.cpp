/**
 * @file
 */
#include "src/statement/LocalVariable.hpp"

#include "src/visitor/all.hpp"

birch::LocalVariable::LocalVariable(const Annotation annotation, Name* name,
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

birch::LocalVariable::LocalVariable(Expression* value, Location* loc) :
    LocalVariable(birch::AUTO, new Name(), new EmptyType(),
    new EmptyExpression(), new EmptyExpression(), value, loc) {
  //
}

birch::LocalVariable::LocalVariable(Name* name, Type* type, Location* loc) :
    LocalVariable(birch::AUTO, name, type, new EmptyExpression(),
    new EmptyExpression(), new EmptyExpression(), loc) {
  //
}

birch::LocalVariable::~LocalVariable() {
  //
}

bool birch::LocalVariable::isDeclaration() const {
  return true;
}

birch::Statement* birch::LocalVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::LocalVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::LocalVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
