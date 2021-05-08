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
    Braced(braces) {
  //
}

void birch::LambdaFunction::accept(Visitor* visitor) const {
  visitor->visit(this);
}
