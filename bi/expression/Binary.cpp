/**
 * @file
 */
#include "bi/expression/Binary.hpp"

#include "bi/visitor/all.hpp"

bi::Binary::Binary(Expression* left, Expression* right, Location* loc) :
    Expression(loc),
    Couple<Expression>(left, right) {
  //
}

bi::Binary::~Binary() {
  //
}

bi::Expression* bi::Binary::getLeft() const {
  return left;
}

bi::Expression* bi::Binary::getRight() const {
  return right;
}

bi::Expression* bi::Binary::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Binary::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Binary::accept(Visitor* visitor) const {
  visitor->visit(this);
}
