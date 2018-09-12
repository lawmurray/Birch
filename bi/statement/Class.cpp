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

bool bi::Class::isBound() const {
  for (auto param : *typeParams) {
    if (param->type->isEmpty()) {
      return false;
    }
  }
  return true;
}

bool bi::Class::isInstantiation() const {
  return isGeneric() && isBound();
}

void bi::Class::bind(Type* typeArgs) {
  assert(typeArgs->width() == typeParams->width());

  Cloner cloner;
  auto arg = typeArgs->begin();
  auto param = typeParams->begin();
  while (arg != typeArgs->end() && param != typeParams->end()) {
    (*param)->type = (*arg)->canonical()->accept(&cloner);
    ++arg;
    ++param;
  }
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
