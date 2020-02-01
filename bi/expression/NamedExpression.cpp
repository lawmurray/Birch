/**
 * @file
 */
#include "bi/expression/NamedExpression.hpp"

#include "bi/visitor/all.hpp"

bi::NamedExpression::NamedExpression(Name* name, Type* typeArgs,
    Location* loc) :
    Expression(loc),
    Named(name),
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
  return true;
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
