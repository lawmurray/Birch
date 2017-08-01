/**
 * @file
 */
#include "bi/statement/Coroutine.hpp"

#include "bi/visitor/all.hpp"

bi::Coroutine::Coroutine(Name* name, Expression* params,
    Type* returnType, Statement* braces, Location* loc) :
    Statement(loc),
    Named(name),
    Parameterised(params),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

bi::Coroutine::~Coroutine() {
  //
}

bi::Statement* bi::Coroutine::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Coroutine::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Coroutine::accept(Visitor* visitor) const {
  visitor->visit(this);
}
