/**
 * @file
 */
#include "bi/expression/VarReference.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::VarReference::VarReference(shared_ptr<Name> name,
    shared_ptr<Location> loc, VarParameter* target) :
    Expression(loc),
    Named(name),
    Reference(target) {
  //
}

bi::VarReference::~VarReference() {
  //
}

bi::Expression* bi::VarReference::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::VarReference::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::VarReference::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::VarReference::dispatchDefinitely(Expression& o) {
  return o.definitely(*this);
}

bool bi::VarReference::definitely(VarReference& o) {
  return target == o.target;
}

bool bi::VarReference::definitely(VarParameter& o) {
  if (!target) {
    return o.capture(this);
  } else {
    return type->definitely(*o.type) && o.capture(this);
  }
}

bool bi::VarReference::dispatchPossibly(Expression& o) {
  return o.possibly(*this);
}

bool bi::VarReference::possibly(VarReference& o) {
  return target == o.target;
}

bool bi::VarReference::possibly(VarParameter& o) {
  if (!target) {
    return o.capture(this);
  } else {
    return type->possibly(*o.type) && o.capture(this);
  }
}
