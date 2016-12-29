/**
 * @file
 */
#include "bi/program/ProgParameter.hpp"

#include "bi/program/ProgReference.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ProgParameter::ProgParameter(shared_ptr<Name> name, Expression* parens,
    Expression* braces, shared_ptr<Location> loc) :
    Prog(loc), Named(name), Parenthesised(parens), Braced(braces) {
  this->arg = this;
}

bi::Prog* bi::ProgParameter::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

void bi::ProgParameter::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::ProgParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ProgParameter::operator<=(Prog& o) {
  try {
    ProgParameter& o1 = dynamic_cast<ProgParameter&>(o);
    return *parens <= *o1.parens && *braces <= *o1.braces && o1.capture(this);
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::ProgParameter::operator==(const Prog& o) const {
  try {
    const ProgParameter& o1 = dynamic_cast<const ProgParameter&>(o);
    return *parens == *o1.parens && *braces == *o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
