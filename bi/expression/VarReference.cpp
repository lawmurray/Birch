/**
 * @file
 */
#include "bi/expression/VarReference.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::VarReference::VarReference(shared_ptr<Name> name,
    shared_ptr<Location> loc, const Parameter* target) :
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

bool bi::VarReference::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::VarReference::definitely(const VarReference& o) const {
  return target == o.target;
}

bool bi::VarReference::definitely(const Parameter& o) const {
  if (!target) {
    return true;
  } else {
    return type->definitely(*o.type);
  }
}

bool bi::VarReference::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::VarReference::possibly(const VarReference& o) const {
  return target == o.target;
}

bool bi::VarReference::possibly(const Parameter& o) const {
  if (!target) {
    return true;
  } else {
    return type->possibly(*o.type);
  }
}
