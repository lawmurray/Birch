/**
 * @file
 */
#include "bi/expression/LambdaInit.hpp"

#include "bi/visitor/all.hpp"

bi::LambdaInit::LambdaInit(Expression* single, shared_ptr<Location> loc) :
    Expression(loc),
    ExpressionUnary(single) {
  //
}

bi::LambdaInit::~LambdaInit() {
  //
}

bi::Expression* bi::LambdaInit::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::LambdaInit::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::LambdaInit::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::LambdaInit::dispatchDefinitely(Expression& o) {
  return o.definitely(*this);
}

bool bi::LambdaInit::definitely(LambdaInit& o) {
  return single->definitely(*o.single);
}

bool bi::LambdaInit::definitely(VarParameter& o) {
  return type->definitely(*o.type) && o.capture(this);
}

bool bi::LambdaInit::dispatchPossibly(Expression& o) {
  return o.possibly(*this);
}

bool bi::LambdaInit::possibly(LambdaInit& o) {
  return single->possibly(*o.single);
}

bool bi::LambdaInit::possibly(VarParameter& o) {
  return type->possibly(*o.type) && o.capture(this);
}
