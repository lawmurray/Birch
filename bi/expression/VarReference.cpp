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

bi::possibly bi::VarReference::dispatch(Expression& o) {
  return o.le(*this);
}

bi::possibly bi::VarReference::le(VarReference& o) {
  return possibly(target != nullptr) && possibly(target == o.target);
}

bi::possibly bi::VarReference::le(VarParameter& o) {
  if (!target) {
    /* not yet bound */
    return o.capture(this);
  } else {
    return *type <= *o.type && o.capture(this);
  }
}
