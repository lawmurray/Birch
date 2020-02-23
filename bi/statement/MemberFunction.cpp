/**
 * @file
 */
#include "bi/statement/MemberFunction.hpp"

#include "bi/visitor/all.hpp"

bi::MemberFunction::MemberFunction(const Annotation annotation, Name* name,
    Expression* typeParams, Expression* params, Type* returnType,
    Statement* braces, Location* loc) :
    Function(annotation, name, typeParams, params, returnType, braces, loc) {
  //
}

bi::MemberFunction::~MemberFunction() {
  //
}

bool bi::MemberFunction::isMember() const {
  return true;
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
