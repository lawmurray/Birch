/**
 * @file
 */
#include "bi/statement/MemberFunction.hpp"

#include "bi/visitor/all.hpp"

bi::MemberFunction::MemberFunction(shared_ptr<Name> name, Expression* parens,
    Type* returnType, Statement* braces, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Parenthesised(parens),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

bi::MemberFunction::~MemberFunction() {
  //
}

bi::Statement* bi::MemberFunction::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::MemberFunction::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::MemberFunction::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::MemberFunction::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::MemberFunction::definitely(const MemberFunction& o) const {
  return parens->definitely(*o.parens);
}

bool bi::MemberFunction::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::MemberFunction::possibly(const MemberFunction& o) const {
  return parens->possibly(*o.parens);
}
