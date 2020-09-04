/**
 * @file
 */
#include "src/expression/Spin.hpp"

#include "src/visitor/all.hpp"

birch::Spin::Spin(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

birch::Spin::~Spin() {
  //
}

bool birch::Spin::isAssignable() const {
  return single->isAssignable();
}

birch::Expression* birch::Spin::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Expression* birch::Spin::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Spin::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
