/**
 * @file
 */
#include "src/statement/MemberFunction.hpp"

#include "src/visitor/all.hpp"

birch::MemberFunction::MemberFunction(const Annotation annotation, Name* name,
    Expression* typeParams, Expression* params, Type* returnType,
    Statement* braces, Location* loc) :
    Function(annotation, name, typeParams, params, returnType, braces, loc) {
  //
}

birch::Statement* birch::MemberFunction::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::MemberFunction::accept(Visitor* visitor) const {
  visitor->visit(this);
}
