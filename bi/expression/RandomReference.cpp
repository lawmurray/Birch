/**
 * @file
 */
#include "bi/expression/RandomReference.hpp"

#include "bi/expression/RandomParameter.hpp"
#include "bi/primitive/encode.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::RandomReference::RandomReference(shared_ptr<Name> name,
    shared_ptr<Location> loc, const RandomParameter* target) :
    Expression(loc),
    Named(name),
    Reference(target) {
  right = new RandomRight(name);
}

bi::RandomReference::RandomReference(Expression* expr) {
  name = new Name(uniqueName(expr));
  right = new RandomRight(name);
}

bi::RandomReference::~RandomReference() {
  //
}

bi::Expression* bi::RandomReference::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::RandomReference::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::RandomReference::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::RandomReference::dispatch(Expression& o) {
  return o.le(*this);
}

bool bi::RandomReference::le(RandomReference& o) {
  return target && target == o.target;
}

bool bi::RandomReference::le(RandomParameter& o) {
  if (!target) {
    /* not yet bound */
    return o.capture(this);
  } else {
    return *target->left <= *o.left && *right <= *o.right
        && o.capture(this);
  }
}

bool bi::RandomReference::le(VarParameter& o) {
  return *type <= *o.type && o.capture(this);
}
