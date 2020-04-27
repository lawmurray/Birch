/**
 * @file
 */
#include "bi/expression/NamedExpression.hpp"

#include "bi/visitor/all.hpp"

bi::NamedExpression::NamedExpression(Name* name, Type* typeArgs,
    Location* loc) :
    Expression(loc),
    Named(name),
    Typed(new EmptyType()),
    TypeArgumented(typeArgs),
    category(UNKNOWN),
    number(0) {
  //
}

bi::NamedExpression::NamedExpression(Name* name, Location* loc) :
    NamedExpression(name, new EmptyType(loc), loc) {
  //
}

bi::NamedExpression::~NamedExpression() {
  //
}

bool bi::NamedExpression::isAssignable() const {
  return category == MEMBER_VARIABLE || category == LOCAL_VARIABLE;
}

bool bi::NamedExpression::isGlobal() const {
  return category == GLOBAL_VARIABLE || category == GLOBAL_FUNCTION ||
      category == GLOBAL_FIBER || category == BINARY_OPERATOR ||
      category == UNARY_OPERATOR;
}

bool bi::NamedExpression::isMember() const {
  return category == MEMBER_VARIABLE || category == MEMBER_FUNCTION ||
      category == MEMBER_FIBER || category == MEMBER_UNKNOWN;
}

bool bi::NamedExpression::isLocal() const {
  return category == LOCAL_VARIABLE;
}

bool bi::NamedExpression::isParameter() const {
  return category == PARAMETER;
}

bi::Expression* bi::NamedExpression::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::NamedExpression::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::NamedExpression::accept(Visitor* visitor) const {
  visitor->visit(this);
}
