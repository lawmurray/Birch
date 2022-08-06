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

void birch::TupleVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
