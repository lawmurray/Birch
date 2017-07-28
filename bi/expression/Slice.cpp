/**
 * @file
 */
#include "bi/expression/Slice.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

bi::Slice::Slice(Expression* single,
    Expression* brackets, shared_ptr<Location> loc) :
    Expression(loc), Unary<Expression>(single), Bracketed(brackets) {
  //
}

bi::Slice::~Slice() {
  //
}

bi::Expression* bi::Slice::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Slice::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Slice::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::Slice::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::Slice::definitely(const Slice& o) const {
  return single->definitely(*o.single) && brackets->definitely(*o.brackets);
}

bool bi::Slice::definitely(const Parameter& o) const {
  return type->definitely(*o.type);
}

bool bi::Slice::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::Slice::possibly(const Slice& o) const {
  return single->possibly(*o.single) && brackets->possibly(*o.brackets);
}

bool bi::Slice::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}
