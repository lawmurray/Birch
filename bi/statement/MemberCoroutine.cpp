/**
 * @file
 */
#include "bi/statement/MemberCoroutine.hpp"

#include "bi/visitor/all.hpp"

bi::MemberCoroutine::MemberCoroutine(shared_ptr<Name> name,
    Expression* parens, Type* returnType, Statement* braces,
    shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Parenthesised(parens),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

bi::MemberCoroutine::~MemberCoroutine() {
  //
}

bi::Statement* bi::MemberCoroutine::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::MemberCoroutine::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::MemberCoroutine::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::MemberCoroutine::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::MemberCoroutine::definitely(const MemberCoroutine& o) const {
  return parens->definitely(*o.parens);
}

bool bi::MemberCoroutine::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::MemberCoroutine::possibly(const MemberCoroutine& o) const {
  return parens->possibly(*o.parens);
}
