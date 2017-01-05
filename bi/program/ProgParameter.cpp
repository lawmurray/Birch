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

bi::Prog* bi::ProgParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Prog* bi::ProgParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ProgParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ProgParameter::dispatch(Prog& o) {
  return o.le(*this);
}

bool bi::ProgParameter::le(ProgParameter& o) {
  return *parens <= *o.parens && *braces <= *o.braces && o.capture(this);
}
