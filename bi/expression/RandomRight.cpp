/**
 * @file
 */
#include "bi/expression/RandomRight.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::RandomRight::RandomRight(shared_ptr<Name> name, shared_ptr<Location> loc) :
    Expression(loc),
    Named(name) {
  //
}

bi::RandomRight::~RandomRight() {
  //
}

bi::Expression* bi::RandomRight::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::RandomRight::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::RandomRight::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::RandomRight::dispatch(Expression& o) {
  return o.le(*this);
}

bool bi::RandomRight::le(RandomRight& o) {
  return *type <= *o.type;
}

bool bi::RandomRight::le(VarParameter& o) {
  return *type <= *o.type && o.capture(this);
}
