/**
 * @file
 */
#include "bi/statement/Class.hpp"

#include "bi/visitor/all.hpp"

bi::Class::Class(Name* name, Expression* typeParams, Expression* params,
    Type* base, const bool alias, Expression* args, Statement* braces,
    Location* loc) :
    Statement(loc),
    Named(name),
    TypeParameterised(typeParams),
    Parameterised(params),
    Based(base, alias),
    Argumented(args),
    Scoped(CLASS_SCOPE),
    Braced(braces),
    initScope(new Scope(LOCAL_SCOPE)),
    state(CLONED) {
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
}

bool bi::Class::hasSuper(const Type* o) const {
  bool result = supers.find(o->canonical()->getClass()) != supers.end();
  result = result || std::any_of(supers.begin(), supers.end(),
      [&](auto x) {return x->hasSuper(o);});
  return result;
}

void bi::Class::addConversion(const Type* o) {
  bool result = std::any_of(conversions.begin(), conversions.end(),
      [&](auto x) {return x->equals(*o);});
  if (!result) {
    conversions.push_back(o);
    if (o->isBasic()) {
      addConversion(o->getBasic()->base);
    } else if (o->isClass()) {
      addConversion(o->getClass()->base);
    }
  }
}

bool bi::Class::hasConversion(const Type* o) const {
  return std::any_of(conversions.begin(), conversions.end(),
      [&](auto x) {return x->equals(*o);}) ||
  std::any_of(supers.begin(), supers.end(),
      [&](auto x) {return x->hasConversion(o);});
}

void bi::Class::addAssignment(const Type* o) {
  bool result = std::any_of(assignments.begin(), assignments.end(),
      [&](auto x) {return x->equals(*o);});
  if (!result) {
    assignments.push_back(o);
    if (o->isBasic()) {
      addAssignment(o->getBasic()->base);
    } else if (o->isClass()) {
      addAssignment(o->getClass()->base);
    }
  }
}

bool bi::Class::hasAssignment(const Type* o) const {
  return std::any_of(assignments.begin(), assignments.end(),
      [&](auto x) {return x->equals(*o);}) ||
  std::any_of(supers.begin(), supers.end(),
      [&](auto x) {return x->hasAssignment(o);});
}
