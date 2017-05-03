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

bi::ProgReference::~ProgReference() {
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

bool bi::ProgReference::dispatchDefinitely(const Prog& o) const {
  return o.definitely(*this);
}

bool bi::ProgReference::definitely(const ProgParameter& o) const {
  return parens->definitely(*o.parens);
}

bool bi::ProgReference::definitely(const ProgReference& o) const {
  return parens->definitely(*o.parens) && target == o.target;
}

bool bi::ProgReference::dispatchPossibly(const Prog& o) const {
  return o.possibly(*this);
}

bool bi::ProgReference::possibly(const ProgParameter& o) const {
  return parens->possibly(*o.parens);
}

bool bi::ProgReference::possibly(const ProgReference& o) const {
  return parens->possibly(*o.parens) && target == o.target;
}
