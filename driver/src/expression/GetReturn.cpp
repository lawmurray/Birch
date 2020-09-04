/**
 * @file
 */
#include "src/expression/GetReturn.hpp"

#include "src/visitor/all.hpp"

birch::GetReturn::GetReturn(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

birch::GetReturn::~GetReturn() {
  //
}

bool birch::GetReturn::isAssignable() const {
  return single->isAssignable();
}

birch::Expression* birch::GetReturn::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Expression* birch::GetReturn::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::GetReturn::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
