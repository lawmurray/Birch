/**
 * @file
 */
#include "bi/expression/Spin.hpp"

#include "bi/visitor/all.hpp"

bi::Spin::Spin(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

bi::Spin::~Spin() {
  //
}

bool bi::Spin::isAssignable() const {
  return single->isAssignable();
}

bi::Expression* bi::Spin::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Spin::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Spin::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
