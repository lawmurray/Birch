/**
 * @file
 */
#include "bi/expression/RandomParameter.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"
#include "bi/primitive/encode.hpp"

#include <typeinfo>

bi::RandomParameter::RandomParameter(Expression* left, Expression* right,
    shared_ptr<Location> loc) :
    Expression(loc),
    ExpressionBinary(left, right) {
  name = new Name(uniqueName(left));
}

bi::RandomParameter::RandomParameter(FuncReference* ref) :
    Expression(ref->loc),
    ExpressionBinary(ref->releaseLeft(), ref->releaseRight()) {
  name = new Name(uniqueName(left.get()));
}

bi::RandomParameter::~RandomParameter() {
  //
}

bi::Expression* bi::RandomParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::RandomParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::RandomParameter::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::RandomParameter::dispatch(Expression& o) {
  return o.le(*this);
}

bool bi::RandomParameter::le(RandomParameter& o) {
  return *left <= *o.left && *right <= *o.right;
}

bool bi::RandomParameter::le(VarParameter& o) {
  return *type <= *o.type && o.capture(this);
}
