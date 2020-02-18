/**
 * @file
 */
#include "bi/statement/MemberFiber.hpp"

#include "bi/visitor/all.hpp"

bi::MemberFiber::MemberFiber(const Annotation annotation, Name* name,
    Expression* typeParams, Expression* params, Type* returnType,
    Statement* braces, Location* loc) :
    Fiber(annotation, name, typeParams, params, returnType, braces, loc) {
  //
}

bi::MemberFiber::~MemberFiber() {
  //
}

bi::Statement* bi::MemberFiber::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::MemberFiber::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::MemberFiber::accept(Visitor* visitor) const {
  visitor->visit(this);
}
