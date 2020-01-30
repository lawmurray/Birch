/**
 * @file
 */
#include "bi/statement/UnaryOperator.hpp"

#include "bi/visitor/all.hpp"

bi::UnaryOperator::UnaryOperator(const Annotation annotation, Name* name,
    Expression* single, Type* returnType, Statement* braces, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    Single<Expression>(single),
    ReturnTyped(returnType),
    Scoped(LOCAL_SCOPE),
    Braced(braces) {
  //
}

bi::UnaryOperator::~UnaryOperator() {
  //
}

bi::Statement* bi::UnaryOperator::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::UnaryOperator::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::UnaryOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}
