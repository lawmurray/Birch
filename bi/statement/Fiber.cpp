/**
 * @file
 */
#include "bi/statement/Fiber.hpp"

#include "bi/visitor/all.hpp"

bi::Fiber::Fiber(const Annotation annotation, Name* name, Expression* params,
    Type* returnType, Statement* braces, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    Parameterised(params),
    ReturnTyped(returnType),
    Typed(new EmptyType(loc)),
    Braced(braces) {
  //
}

bi::Fiber::~Fiber() {
  //
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
