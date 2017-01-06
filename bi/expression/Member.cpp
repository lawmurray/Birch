/**
 * @file
 */
#include "bi/expression/Member.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Member::Member(Expression* left, Expression* right,
    shared_ptr<Location> loc) :
    Expression(loc),
    ExpressionBinary(left, right) {
  //
}

bi::Member::~Member() {
  //
}

bi::Expression* bi::Member::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Member::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Member::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::Member::dispatch(Expression& o) {
  return o.le(*this);
}

bool bi::Member::le(Member& o) {
  return *left <= *o.left && *right <= *o.right;
}

bool bi::Member::le(VarParameter& o) {
  return *type <= *o.type && o.capture(this);
}
