/**
 * @file
 */
#include "bi/expression/GetReturn.hpp"

#include "bi/visitor/all.hpp"

bi::GetReturn::GetReturn(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

bi::GetReturn::~GetReturn() {
  //
}

bool bi::GetReturn::isAssignable() const {
  return single->isAssignable();
}

bi::Expression* bi::GetReturn::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::GetReturn::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::GetReturn::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
