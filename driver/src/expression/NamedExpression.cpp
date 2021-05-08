/**
 * @file
 */
#include "src/expression/NamedExpression.hpp"

#include "src/visitor/all.hpp"

birch::NamedExpression::NamedExpression(Name* name, Type* typeArgs,
    Location* loc) :
    Expression(loc),
    Named(name),
    Typed(new EmptyType()),
    TypeArgumented(typeArgs) {
  //
}

birch::NamedExpression::NamedExpression(Name* name, Location* loc) :
    NamedExpression(name, new EmptyType(loc), loc) {
  //
}

birch::Expression* birch::NamedExpression::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::NamedExpression::accept(Visitor* visitor) const {
  visitor->visit(this);
}
