/**
 * @file
 */
#include "bi/statement/Fiber.hpp"

#include "bi/visitor/all.hpp"

bi::Fiber::Fiber(const Annotation annotation, Name* name,
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
  if (!returnType->isFiber()) {
    this->returnType = new FiberType(returnType, new EmptyType(loc), loc);
  }
}

bi::Fiber::~Fiber() {
  //
}

bool bi::Fiber::isDeclaration() const {
  return true;
}

bool bi::Fiber::isMember() const {
  return true;
}

bi::Statement* bi::Fiber::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Fiber::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Fiber::accept(Visitor* visitor) const {
  visitor->visit(this);
}
