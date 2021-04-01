/**
 * @file
 */
#include "src/statement/MemberVariable.hpp"

#include "src/visitor/all.hpp"

birch::MemberVariable::MemberVariable(const Annotation annotation, Name* name,
    Type* type, Expression* brackets, Expression* args, Name* op,
    Expression* value, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    Typed(type),
    Bracketed(brackets),
    Argumented(args),
    Valued(op, value) {
  assert(value->isEmpty() || args->isEmpty());
}

birch::MemberVariable::~MemberVariable() {
  //
}

birch::Statement* birch::MemberVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::MemberVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::MemberVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
