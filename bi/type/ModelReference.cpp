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

  if (*target->op == "=") {
    return target->base->isBuiltin();
  } else {
    return target->braces->isEmpty();
  }
}

bool bi::ModelReference::isModel() const {
  /* pre-condition */
  assert(target);

  if (*target->op == "=") {
    return target->base->isModel();
  } else {
    return !target->braces->isEmpty();
  }
}

bi::possibly bi::ModelReference::isa(ModelReference& o) {
  bool result = target == o.target;
  if (!result && target) {
    ModelReference* ref = dynamic_cast<ModelReference*>(target->base.get());
    result = ref && ref->isa(o);
  }
  return possibly(result);
}

bi::possibly bi::ModelReference::dispatch(Type& o) {
  return o.le(*this);
}

bi::possibly bi::ModelReference::le(ModelParameter& o) {
  if (!target) {
    /* not yet bound */
    return o.capture(this);
  } else {
    return *this <= *o.base && o.capture(this);
  }
}

bi::possibly bi::ModelReference::le(ModelReference& o) {
  if (*o.target->op == "=") {
    return *this <= *o.target->base;  // compare with canonical type
  } else {
    return (isa(o) || (possible && o.isa(*this))) && *parens <= *o.parens
        && (!o.assignable || assignable);
  }
}

bi::possibly bi::ModelReference::le(AssignableType& o) {
  return *this <= *o.single;
}

bi::possibly bi::ModelReference::le(ParenthesesType& o) {
  return *this <= *o.single;
}

bi::possibly bi::ModelReference::le(EmptyType& o) {
  return possibly(!o.assignable || assignable);
}
