/**
 * @file
 */
#include "src/statement/MemberFiber.hpp"

#include "src/visitor/all.hpp"

birch::MemberFiber::MemberFiber(const Annotation annotation, Name* name,
    Expression* typeParams, Expression* params, Type* returnType,
    Statement* braces, Location* loc) :
    Fiber(annotation, name, typeParams, params, returnType, braces, loc) {
  //
}

birch::MemberFiber::~MemberFiber() {
  //
}

bool birch::MemberFiber::isMember() const {
  return true;
}

birch::Statement* birch::MemberFiber::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::MemberFiber::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::MemberFiber::accept(Visitor* visitor) const {
  visitor->visit(this);
}
