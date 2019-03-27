/**
 * @file
 */
#include "bi/statement/Class.hpp"

#include "bi/visitor/all.hpp"

bi::Class::Class(const Annotation annotation, Name* name,
    Expression* typeParams, Expression* params, Type* base, const bool alias,
    Expression* args, Statement* braces,
    Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    TypeParameterised(typeParams),
    Parameterised(params),
    Based(base, alias),
    Argumented(args),
    Scoped(CLASS_SCOPE),
    Braced(braces),
    initScope(new Scope(LOCAL_SCOPE)) {
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
  supers.insert(o->getClass());
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
  }
}

bool bi::Class::hasConversion(const Type* o) const {
  return std::any_of(conversions.begin(), conversions.end(),
      [&](auto x) {return x->definitely(*o);}) ||
  std::any_of(supers.begin(), supers.end(),
      [&](auto x) {return x->hasConversion(o);});
}

void bi::Class::addAssignment(const Type* o) {
  bool result = std::any_of(assignments.begin(), assignments.end(),
      [&](auto x) {return x->equals(*o);});
  if (!result) {
    assignments.push_back(o);
  }
}

bool bi::Class::hasAssignment(const Type* o) const {
  return std::any_of(assignments.begin(), assignments.end(),
      [&](auto x) {return o->definitely(*x);}) ||
  std::any_of(supers.begin(), supers.end(),
      [&](auto x) {return x->hasAssignment(o);});
}
