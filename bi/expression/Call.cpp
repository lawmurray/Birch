/**
 * @file
 */
#include "bi/expression/Call.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

bi::Call::Call(Expression* single, Expression* parens,
    shared_ptr<Location> loc) :
    Expression(loc),
    Unary<Expression>(single),
    Parenthesised(parens) {
  //
}

bi::Call::~Call() {
  //
}

bi::Expression* bi::Call::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Call::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Call::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::Call::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::Call::definitely(const Call& o) const {
  return single->definitely(*o.single) && parens->definitely(*o.parens);
}

bool bi::Call::definitely(const Parameter& o) const {
  return type->definitely(*o.type);
}

bool bi::Call::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::Call::possibly(const Call& o) const {
  return single->possibly(*o.single) && parens->possibly(*o.parens);
}

bool bi::Call::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}
