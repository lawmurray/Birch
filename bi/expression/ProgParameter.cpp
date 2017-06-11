/**
 * @file
 */
#include "bi/expression/ProgParameter.hpp"

#include "bi/expression/ProgReference.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ProgParameter::ProgParameter(shared_ptr<Name> name, Expression* parens,
    Expression* braces, shared_ptr<Location> loc) :
    Expression(loc),
    Named(name),
    Parenthesised(parens),
    Braced(braces) {
  //
}

bi::ProgParameter::~ProgParameter() {
  //
}

bi::Expression* bi::ProgParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::ProgParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ProgParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ProgParameter::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::ProgParameter::definitely(const ProgParameter& o) const {
  return parens->definitely(*o.parens) && braces->definitely(*o.braces);
}

bool bi::ProgParameter::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::ProgParameter::possibly(const ProgParameter& o) const {
  return parens->possibly(*o.parens) && braces->possibly(*o.braces);
}
