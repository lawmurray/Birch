/**
 * @file
 */
#include "src/statement/Fiber.hpp"

#include "src/visitor/all.hpp"

birch::Fiber::Fiber(const Annotation annotation, Name* name,
    Expression* typeParams, Expression* params, Type* returnType,
    Statement* braces, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    TypeParameterised(typeParams),
    Parameterised(params),
    ReturnTyped(returnType),
    Scoped(LOCAL_SCOPE),
    Braced(braces),
    start(nullptr) {
  //
}

birch::Fiber::~Fiber() {
  //
}

bool birch::Fiber::isDeclaration() const {
  return true;
}

bool birch::Fiber::isMember() const {
  return true;
}

birch::Statement* birch::Fiber::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::Fiber::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Fiber::accept(Visitor* visitor) const {
  visitor->visit(this);
}
