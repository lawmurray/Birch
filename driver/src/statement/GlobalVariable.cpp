/**
 * @file
 */
#include "src/statement/GlobalVariable.hpp"

#include "src/visitor/all.hpp"

birch::GlobalVariable::GlobalVariable(const Annotation annotation, Name* name, Type* type,
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

birch::GlobalVariable::~GlobalVariable() {
  //
}

bool birch::GlobalVariable::isDeclaration() const {
  return true;
}

birch::Statement* birch::GlobalVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::GlobalVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::GlobalVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
