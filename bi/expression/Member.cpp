/**
 * @file
 */
#include "bi/expression/Member.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Member::Member(Expression* left, Expression* right,
    shared_ptr<Location> loc) :
    Expression(loc),
    Binary<Expression>(left, right) {
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

bool bi::Member::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::Member::definitely(const Member& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::Member::definitely(const Parameter& o) const {
  return type->definitely(*o.type);
}

bool bi::Member::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::Member::possibly(const Member& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}

bool bi::Member::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}
