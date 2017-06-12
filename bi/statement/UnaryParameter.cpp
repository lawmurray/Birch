/**
 * @file
 */
#include "bi/statement/UnaryParameter.hpp"

#include "bi/visitor/all.hpp"

bi::UnaryParameter::UnaryParameter(shared_ptr<Name> name, Expression* single,
    Type* type, Expression* braces, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Unary(single),
    Typed(type),
    Braced(braces) {
  //
}

bi::UnaryParameter::~UnaryParameter() {
  //
}

bi::Statement* bi::UnaryParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::UnaryParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::UnaryParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::UnaryParameter::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::UnaryParameter::definitely(const UnaryParameter& o) const {
  return single->definitely(*o.single);
}

bool bi::UnaryParameter::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::UnaryParameter::possibly(const UnaryParameter& o) const {
  return single->possibly(*o.single);
}
