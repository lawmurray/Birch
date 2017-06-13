/**
 * @file
 */
#include "bi/expression/Range.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Range::Range(Expression* left, Expression* right,
    shared_ptr<Location> loc) :
    Expression(loc),
    Binary<Expression>(left, right) {
  //
}

bi::Range::~Range() {
  //
}

bi::Expression* bi::Range::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Range::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Range::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::Range::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::Range::definitely(const Range& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::Range::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::Range::possibly(const Range& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}
