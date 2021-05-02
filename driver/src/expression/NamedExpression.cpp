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
    TypeArgumented(typeArgs),
    uses(nullptr),
    rank(0),
    number(0),
    category(UNKNOWN) {
  //
}

birch::NamedExpression::NamedExpression(Name* name, Location* loc) :
    NamedExpression(name, new EmptyType(loc), loc) {
  //
}

bool birch::NamedExpression::isAssignable() const {
  return category == MEMBER_VARIABLE || category == LOCAL_VARIABLE;
}

bool birch::NamedExpression::isGlobal() const {
  return category == GLOBAL_VARIABLE || category == GLOBAL_FUNCTION ||
      category == BINARY_OPERATOR || category == UNARY_OPERATOR;
}

bool birch::NamedExpression::isMember() const {
  return category == MEMBER_VARIABLE || category == MEMBER_FUNCTION ||
      category == MEMBER_UNKNOWN;
}

bool birch::NamedExpression::isLocal() const {
  return category == LOCAL_VARIABLE;
}

bool birch::NamedExpression::isParameter() const {
  return category == PARAMETER;
}

birch::Expression* birch::NamedExpression::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::NamedExpression::accept(Visitor* visitor) const {
  visitor->visit(this);
}
