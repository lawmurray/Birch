/**
 * @file
 */
#include "bi/program/ProgReference.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ProgReference::ProgReference(shared_ptr<Name> name, Expression* parens,
    shared_ptr<Location> loc, const ProgParameter* target) :
    Prog(loc), Named(name), Parenthesised(parens), Reference(target) {
  //
}

bi::Prog* bi::ProgReference::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

void bi::ProgReference::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::ProgReference::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ProgReference::operator<=(Prog& o) {
  if (!target) {
    /* not yet bound */
    try {
      ProgParameter& o1 = dynamic_cast<ProgParameter&>(o);
      return *parens <= *o1.parens;
    } catch (std::bad_cast e) {
      //
    }
  } else {
    try {
      ProgReference& o1 = dynamic_cast<ProgReference&>(o);
      return *parens <= *o1.parens && (o1.canon(this) || o1.check(this));
    } catch (std::bad_cast e) {
      //
    }
    try {
      ProgParameter& o1 = dynamic_cast<ProgParameter&>(o);
      return *parens <= *o1.parens && o1.capture(this);
    } catch (std::bad_cast e) {
      //
    }
  }
  return false;
}

bool bi::ProgReference::operator==(const Prog& o) const {
  try {
    const ProgReference& o1 = dynamic_cast<const ProgReference&>(o);
    return *parens == *o1.parens && o1.canon(this);
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
