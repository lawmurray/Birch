/**
 * @file
 */
#include "bi/program/ProgReference.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ProgReference::ProgReference(shared_ptr<Name> name, Expression* parens,
    shared_ptr<Location> loc, const ProgParameter* target) :
    Prog(loc),
    Named(name),
    Parenthesised(parens),
    Reference(target) {
  //
}

bi::Prog* bi::ProgReference::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Prog* bi::ProgReference::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ProgReference::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ProgReference::dispatch(Prog& o) {
  return o.le(*this);
}

bool bi::ProgReference::le(ProgParameter& o) {
  if (!target) {
    /* not yet bound */
    return *parens <= *o.parens;
  } else {
    return *parens <= *o.parens && o.capture(this);
  }
}

bool bi::ProgReference::le(ProgReference& o) {
  return *parens <= *o.parens && (o.canon(this) || o.check(this));
}
