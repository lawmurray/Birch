/**
 * @file
 */
#include "src/expression/LambdaFunction.hpp"

#include "src/visitor/all.hpp"

birch::LambdaFunction::LambdaFunction(Expression* params, Type* returnType,
    Statement* braces, Location* loc) :
    Expression(loc),
    Parameterised(params),
    ReturnTyped(returnType),
    Scoped(LOCAL_SCOPE),
    Braced(braces) {
  //
}

birch::Expression* birch::LambdaFunction::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::LambdaFunction::accept(Visitor* visitor) const {
  visitor->visit(this);
}
