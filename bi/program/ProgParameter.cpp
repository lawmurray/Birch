/**
 * @file
 */
#include "bi/program/ProgParameter.hpp"

#include "bi/program/ProgReference.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ProgParameter::ProgParameter(shared_ptr<Name> name, Expression* parens,
    Expression* braces, shared_ptr<Location> loc) :
    Prog(loc),
    Named(name),
    Parenthesised(parens),
    Braced(braces) {
  this->arg = this;
}

bi::ProgParameter::~ProgParameter() {
  //
}

bi::Prog* bi::ProgParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Prog* bi::ProgParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ProgParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ProgParameter::dispatchDefinitely(Prog& o) {
  return o.definitely(*this);
}

bool bi::ProgParameter::definitely(ProgParameter& o) {
  return parens->definitely(*o.parens) && braces->definitely(*o.braces)
      && o.capture(this);
}

bool bi::ProgParameter::dispatchPossibly(Prog& o) {
  return o.possibly(*this);
}

bool bi::ProgParameter::possibly(ProgParameter& o) {
  return parens->possibly(*o.parens) && braces->possibly(*o.braces)
      && o.capture(this);
}
