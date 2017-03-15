/**
 * @file
 */
#include "bi/expression/RandomInit.hpp"

#include "bi/visitor/all.hpp"

bi::RandomInit::RandomInit(Expression* left, Expression* right,
    shared_ptr<Location> loc) :
    Expression(loc),
    ExpressionBinary(left, right) {
  //
}

bi::RandomInit::~RandomInit() {
  //
}

bi::Expression* bi::RandomInit::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::RandomInit::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::RandomInit::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::RandomInit::dispatchDefinitely(Expression& o) {
  return o.definitely(*this);
}

bool bi::RandomInit::definitely(RandomInit& o) {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::RandomInit::definitely(VarParameter& o) {
  return type->definitely(*o.type) && o.capture(this);
}

bool bi::RandomInit::dispatchPossibly(Expression& o) {
  return o.possibly(*this);
}

bool bi::RandomInit::possibly(RandomInit& o) {
  return left->possibly(*o.left) && right->possibly(*o.right);
}

bool bi::RandomInit::possibly(VarParameter& o) {
  return type->possibly(*o.type) && o.capture(this);
}
