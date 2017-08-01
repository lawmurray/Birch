/**
 * @file
 */
#include "bi/statement/MemberCoroutine.hpp"

#include "bi/visitor/all.hpp"

bi::MemberCoroutine::MemberCoroutine(Name* name,
    Expression* parens, Type* returnType, Statement* braces,
    Location* loc) :
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
