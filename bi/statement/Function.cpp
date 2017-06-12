/**
 * @file
 */
#include "bi/statement/Function.hpp"

#include "bi/visitor/all.hpp"

bi::Function::Function(shared_ptr<Name> name, Expression* parens, Type* returnType,
    Expression* braces, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Parenthesised(parens),
    ReturnTyped(returnType),
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

bool bi::Function::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::Function::definitely(const Function& o) const {
  return parens->definitely(*o.parens);
}

bool bi::Function::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::Function::possibly(const Function& o) const {
  return parens->possibly(*o.parens);
}
