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
    Scoped(LOCAL_SCOPE),
    Braced(braces) {
  //
}

birch::Function::~Function() {
  //
}

bool birch::Function::isMember() const {
  return false;
}

birch::Statement* birch::Function::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::Function::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Function::accept(Visitor* visitor) const {
  visitor->visit(this);
}
