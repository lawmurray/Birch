/**
 * @file
 */
#include "bi/statement/Function.hpp"

#include "bi/visitor/all.hpp"

bi::Function::Function(const Annotation annotation, Name* name,
    Expression* typeParams, Expression* params, Type* returnType,
    Statement* braces, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    TypeParameterised(typeParams),
    Parameterised(params),
    ReturnTyped(returnType),
    Typed(new EmptyType(loc)),
    Scoped(LOCAL_SCOPE),
    Braced(braces) {
  //
}

bi::Function::~Function() {
  //
}

bi::Statement* bi::Function::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Function::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Function::accept(Visitor* visitor) const {
  visitor->visit(this);
}
