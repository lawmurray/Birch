/**
 * @file
 */
#include "bi/type/ModelReference.hpp"

#include "bi/visitor/all.hpp"

bi::ModelReference::ModelReference(shared_ptr<Name> name, Expression* parens,
    shared_ptr<Location> loc, ModelParameter* target) :
    Type(loc),
    Named(name),
    Parenthesised(parens),
    Reference(target) {
  //
}

bi::ModelReference::ModelReference(ModelParameter* target) :
    Named(target->name),
    Reference(target) {
  //
}

bi::ModelReference::~ModelReference() {
  //
}

bi::Type* bi::ModelReference::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::ModelReference::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ModelReference::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ModelReference::isBuiltin() const {
  /* pre-condition */
  assert(target);

  if (target->isEqual()) {
    return target->base->isBuiltin();
  } else {
    return target->braces->isEmpty();
  }
}

bool bi::ModelReference::isModel() const {
  /* pre-condition */
  assert(target);

  if (target->isEqual()) {
    return target->base->isModel();
  } else {
    return !target->braces->isEmpty();
  }
}

bool bi::ModelReference::canUpcast(ModelReference& o) {
  /* pre-condition */
  assert(target && o.target);

  if (o.target->isEqual()) {
    ModelReference* ref =
        dynamic_cast<ModelReference*>(o.target->base->strip());
    return canUpcast(*ref);  // compare with canonical type
  } else {
    ModelReference* ref = dynamic_cast<ModelReference*>(target->base->strip());
    return target == o.target || (ref && ref->canUpcast(o));
  }
}

bool bi::ModelReference::canDowncast(ModelReference& o) {
  return o.canUpcast(*this);
}

bool bi::ModelReference::dispatchDefinitely(Type& o) {
  return o.definitely(*this);
}

bool bi::ModelReference::definitely(ModelParameter& o) {
  return o.capture(this);
}

bool bi::ModelReference::definitely(ModelReference& o) {
  return canUpcast(o) && (!o.assignable || assignable);
}

bool bi::ModelReference::definitely(LambdaType& o) {
  return definitely(*o.result) && (!o.assignable || assignable);
}

bool bi::ModelReference::definitely(EmptyType& o) {
  return !o.assignable || assignable;
}

bool bi::ModelReference::dispatchPossibly(Type& o) {
  return o.possibly(*this);
}

bool bi::ModelReference::possibly(ModelParameter& o) {
  return o.capture(this);
}

bool bi::ModelReference::possibly(ModelReference& o) {
  return canDowncast(o) && (!o.assignable || assignable);
}

bool bi::ModelReference::possibly(LambdaType& o) {
  return possibly(*o.result) && (!o.assignable || assignable);
}

bool bi::ModelReference::possibly(EmptyType& o) {
  return !o.assignable || assignable;
}
