/**
 * @file
 */
#include "bi/statement/Coroutine.hpp"

#include "bi/visitor/all.hpp"

bi::Coroutine::Coroutine(shared_ptr<Name> name, Expression* parens,
    Type* returnType, Expression* braces, shared_ptr<Location> loc) :
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

bool bi::Coroutine::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::Coroutine::definitely(const Coroutine& o) const {
  return parens->definitely(*o.parens);
}

bool bi::Coroutine::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::Coroutine::possibly(const Coroutine& o) const {
  return parens->possibly(*o.parens);
}
