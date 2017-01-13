/**
 * @file
 */
#include "bi/expression/RandomInit.hpp"

#include "bi/visitor/all.hpp"
#include "bi/primitive/encode.hpp"

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

bool bi::RandomInit::dispatch(Expression& o) {
  return o.le(*this);
}

bool bi::RandomInit::le(RandomInit& o) {
  return *left <= *o.left && *right <= *o.right;
}

bool bi::RandomInit::le(VarParameter& o) {
  return *type <= *o.type && o.capture(this);
}
