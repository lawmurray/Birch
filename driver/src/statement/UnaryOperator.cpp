/**
 * @file
 */
#include "src/statement/UnaryOperator.hpp"

#include "src/visitor/all.hpp"

birch::UnaryOperator::UnaryOperator(const Annotation annotation, Expression* typeParams,
    Name* name, Expression* single, Type* returnType, Statement* braces,
    Location* loc) :
    Statement(loc),
    Annotated(annotation),
    TypeParameterised(typeParams),
    Named(name),
    Single<Expression>(single),
    ReturnTyped(returnType),
    Scoped(LOCAL_SCOPE),
    Braced(braces) {
  //
}

birch::UnaryOperator::~UnaryOperator() {
  //
}

bool birch::UnaryOperator::isDeclaration() const {
  return true;
}

birch::Statement* birch::UnaryOperator::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::UnaryOperator::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::UnaryOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}
