/**
 * @file
 */
#include "src/statement/TupleVariable.hpp"

#include "src/visitor/all.hpp"

birch::TupleVariable::TupleVariable(const Annotation annotation,
    Statement* locals, Name* op, Expression* value, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Valued(op, value),
    locals(locals) {
  //
}

birch::TupleVariable::~TupleVariable() {
  //
}

birch::Statement* birch::TupleVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::TupleVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::TupleVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
