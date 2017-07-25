/**
 * @file
 */
#include "bi/statement/Yield.hpp"

#include "bi/visitor/all.hpp"

bi::Yield::Yield(Expression* single,
    shared_ptr<Location> loc) :
    Statement(loc),
    Unary<Expression>(single) {
  //
}

bi::Yield::~Yield() {
  //
}

bi::Statement* bi::Yield::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Yield::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Yield::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Yield::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::Yield::definitely(const Yield& o) const {
  return single->definitely(*o.single);
}

bool bi::Yield::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::Yield::possibly(const Yield& o) const {
  return single->possibly(*o.single);
}
