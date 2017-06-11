/**
 * @file
 */
#include "bi/expression/UnaryParameter.hpp"

#include "bi/visitor/all.hpp"

bi::UnaryParameter::UnaryParameter(shared_ptr<Name> name, Expression* single,
    Type* type, Expression* braces, shared_ptr<Location> loc) :
    Expression(type, loc),
    Named(name),
    Unary(single),
    Braced(braces) {
  //
}

bi::UnaryParameter::~UnaryParameter() {
  //
}

bi::Expression* bi::UnaryParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::UnaryParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::UnaryParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::UnaryParameter::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::UnaryParameter::definitely(const UnaryParameter& o) const {
  return single->definitely(*o.single);
}

bool bi::UnaryParameter::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::UnaryParameter::possibly(const UnaryParameter& o) const {
  return single->possibly(*o.single);
}
