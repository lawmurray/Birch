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

bool bi::Member::dispatchDefinitely(Expression& o) {
  return o.definitely(*this);
}

bool bi::Member::definitely(Member& o) {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::Member::definitely(VarParameter& o) {
  return type->definitely(*o.type) && o.capture(this);
}

bool bi::Member::dispatchPossibly(Expression& o) {
  return o.possibly(*this);
}

bool bi::Member::possibly(Member& o) {
  return left->possibly(*o.left) && right->possibly(*o.right);
}

bool bi::Member::possibly(VarParameter& o) {
  return type->possibly(*o.type) && o.capture(this);
}
