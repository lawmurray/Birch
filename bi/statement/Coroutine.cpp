/**
 * @file
 */
#include "bi/statement/Coroutine.hpp"

#include "bi/visitor/all.hpp"

bi::Coroutine::Coroutine(shared_ptr<Name> name, Expression* parens,
    Type* returnType, Statement* braces, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Parenthesised(parens),
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
