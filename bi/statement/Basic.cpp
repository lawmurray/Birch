/**
 * @file
 */
#include "bi/statement/Basic.hpp"

#include "bi/visitor/all.hpp"

bi::Basic::Basic(Name* name, Type* base, const bool alias, Location* loc) :
    Statement(loc),
    Named(name),
    Based(base, alias) {
  //
}

bi::Basic::~Basic() {
  //
}

bi::Statement* bi::Basic::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Basic::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Basic::accept(Visitor* visitor) const {
  visitor->visit(this);
}

void bi::Basic::addSuper(const Type* o) {
  auto base = o->getBasic();
  supers.insert(base);
}

bool bi::Basic::hasSuper(const Type* o) const {
  bool result = supers.find(o->getBasic()) != supers.end();
  result = result || std::any_of(supers.begin(), supers.end(),
      [&](auto x) { return x->hasSuper(o); });
  return result;
}
