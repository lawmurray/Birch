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
  //
}

bi::RandomReference::RandomReference(Expression* expr) {
  name = new Name(uniqueName(expr));
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

bool bi::RandomReference::le(RandomParameter& o) {
  return *left <= *o.left && *right <= *o.right && *type <= *o.type
      && o.capture(this);
}

bool bi::RandomReference::le(RandomReference& o) {
  return *type <= *o.type && (o.canon(this) || o.check(this));
}

bool bi::RandomReference::le(VarParameter& o) {
  return *type <= *o.type && o.capture(this);
}

bool bi::RandomReference::le(VarReference& o) {
  return *type <= *o.type && o.check(this);
}
