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
    Parameterised(params),
    Based(base, alias),
    Argumented(args),
    Scoped(CLASS_SCOPE),
    Braced(braces),
    typeParams(typeParams),
    initScope(new Scope(LOCAL_SCOPE)),
    state(CLONED),
    isInstantiation(false),
    isExplicit(false) {
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

bool bi::Class::isGeneric() const {
  return !typeParams->isEmpty();
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
  conversions.push_back(o);
}

bool bi::Class::hasConversion(const Type* o) const {
  bool result = std::any_of(conversions.begin(), conversions.end(),
      [&](auto x) {return x->equals(*o);});
  result = result
      || std::any_of(supers.begin(), supers.end(),
          [&](auto x) {return x->hasConversion(o);});
  return result;
}

void bi::Class::addAssignment(const Type* o) {
  assignments.push_back(o);
}

bool bi::Class::hasAssignment(const Type* o) const {
  bool result = std::any_of(assignments.begin(), assignments.end(),
      [&](auto x) {return x->equals(*o);});
  result = result
      || std::any_of(supers.begin(), supers.end(),
          [&](auto x) {return x->hasAssignment(o);});
  return result;
}

void bi::Class::addInstantiation(Class* o) {
  instantiations.push_back(o);
}

bi::Class* bi::Class::getInstantiation(const Type* typeArgs) {
  auto compare = [](const Type* arg, const Expression* param) {
    return arg->equals(*param->type);
  };
  if (compare(typeArgs, typeParams)) {
    return this;
  } else {
    for (auto o : instantiations) {
      bool matches = typeArgs->width() == o->typeParams->width()
          && std::equal(typeArgs->begin(), typeArgs->end(),
              o->typeParams->begin(), compare);
      if (matches) {
        return o;
      }
    }
  }
  return nullptr;
}
