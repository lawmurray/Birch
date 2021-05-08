/**
 * @file
 */
#include "src/statement/Function.hpp"

#include "src/visitor/all.hpp"

birch::Function::Function(const Annotation annotation, Name* name,
    Expression* typeParams, Expression* params, Type* returnType,
    Statement* braces, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    TypeParameterised(typeParams),
    Parameterised(params),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

birch::Statement* birch::Function::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Function::accept(Visitor* visitor) const {
  visitor->visit(this);
}
