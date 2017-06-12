/**
 * @file
 */
#include "bi/statement/UnaryOperator.hpp"

#include "bi/visitor/all.hpp"

bi::UnaryOperator::UnaryOperator(shared_ptr<Name> name, Expression* single,
    Type* returnType, Expression* braces, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Unary(single),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

bi::UnaryOperator::~UnaryOperator() {
  //
}

bi::Statement* bi::UnaryOperator::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::UnaryOperator::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::UnaryOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::UnaryOperator::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::UnaryOperator::definitely(const UnaryOperator& o) const {
  return single->definitely(*o.single);
}

bool bi::UnaryOperator::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::UnaryOperator::possibly(const UnaryOperator& o) const {
  return single->possibly(*o.single);
}
