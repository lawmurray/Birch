/**
 * @file
 */
#include "bi/type/ModelReference.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ModelReference::ModelReference(shared_ptr<Name> name,
    Expression* brackets, shared_ptr<Location> loc,
    const ModelParameter* target) :
    Type(loc), Named(name), Bracketed(brackets), Reference(target), ndims(
        brackets->tupleSize()) {
  //
}

bi::ModelReference::ModelReference(shared_ptr<Name> name, const int ndims,
    const ModelParameter* target) :
    Named(name), Reference(target), ndims(ndims) {
  //
}

bi::ModelReference::~ModelReference() {
  //
}

bi::Type* bi::ModelReference::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::ModelReference::acceptModify(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ModelReference::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ModelReference::builtin() const {
  /* pre-condition */
  assert(target);

  if (*target->op == "=") {
    return target->base->builtin();
  } else {
    return !*target->braces;
  }
}

int bi::ModelReference::count() const {
  return ndims;
}

bool bi::ModelReference::operator<=(Type& o) {
  if (!target) {
    /* not yet bound */
    try {
      ModelParameter& o1 = dynamic_cast<ModelParameter&>(o);
      return o1.capture(this);
    } catch (std::bad_cast e) {
      //
    }
  } else {
    try {
      ModelReference& o1 = dynamic_cast<ModelReference&>(o);
      if (*o1.target->op == "=") {
        return *this <= *o1.target->base.get() && *brackets <= *o1.brackets/* && ndims == o1.ndims*/;;  // compare with canonical type
      } else {
        return o1.canon(this) || o1.check(this) || *target->base.get() <= o1;
      }
    } catch (std::bad_cast e) {
      //
    }
    try {
      ModelParameter& o1 = dynamic_cast<ModelParameter&>(o);
      return *this <= *o1.base && o1.capture(this);
    } catch (std::bad_cast e) {
      //
    }
    try {
      EmptyType& o1 = dynamic_cast<EmptyType&>(o);
      return true;
    } catch (std::bad_cast e) {
      //
    }
  }
  try {
    ParenthesesType& o1 = dynamic_cast<ParenthesesType&>(o);
    return *this <= *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::ModelReference::operator==(const Type& o) const {
  try {
    const ModelReference& o1 = dynamic_cast<const ModelReference&>(o);
    return o1.canon(this) && *brackets == *o1.brackets/* && ndims == o1.ndims*/;;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
