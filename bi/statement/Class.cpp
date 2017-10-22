/**
 * @file
 */
#include "bi/statement/Class.hpp"

#include "bi/visitor/all.hpp"

bi::Class::Class(Name* name, Expression* params, Type* base,
    Expression* baseArgs, Statement* braces, Location* loc) :
    Statement(loc),
    Named(name),
    Parameterised(params),
    Based(base),
    Braced(braces),
    baseArgs(baseArgs) {
  //
}

bi::Class::~Class() {
  //
}

bi::Statement* bi::Class::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Class::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Class::accept(Visitor* visitor) const {
  visitor->visit(this);
}

void bi::Class::addSuper(const Type* o) {
  auto base = o->getClass();
  supers.insert(base);
  scope->inherit(base->scope);
}

bool bi::Class::hasSuper(const Type* o) const {
  bool result = supers.find(o->getClass()) != supers.end();
  result = result || std::any_of(supers.begin(), supers.end(),
      [&](auto x) { return x->hasSuper(o); });
  return result;
}

void bi::Class::addConversion(const Type* o) {
  conversions.push_back(o);
}

bool bi::Class::hasConversion(const Type* o) const {
  bool result = std::any_of(conversions.begin(), conversions.end(),
      [&](auto x) { return x->equals(*o); });
  result = result || std::any_of(supers.begin(), supers.end(),
      [&](auto x) { return x->hasConversion(o); });
  return result;
}

void bi::Class::addAssignment(const Type* o) {
  assignments.push_back(o);
}

bool bi::Class::hasAssignment(const Type* o) const {
  bool result = std::any_of(assignments.begin(), assignments.end(),
      [&](auto x) { return x->equals(*o); });
  result = result || std::any_of(supers.begin(), supers.end(),
      [&](auto x) { return x->hasAssignment(o); });
  return result;
}
