/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

#include "bi/exception/all.hpp"
#include "bi/visitor/all.hpp"

bi::Resolver::Resolver(Package* currentPackage, Class* currentClass,
    Fiber* currentFiber) :
    ScopedModifier(currentPackage, currentClass, currentFiber) {
  //
}

bi::Resolver::~Resolver() {
  //
}

bi::Expression* bi::Resolver::modify(Parameter* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Resolver::modify(LocalVariable* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Expression* bi::Resolver::modify(NamedExpression* o) {
  ScopedModifier::modify(o);
  if (inMember) {
    /* Clearly a member something, but cannot determine whether this
     * something is a variable, function or fiber without type deduction.
     * Instead categorize as MEMBER_UNKNOWN, and let C++ handle the rest */
    o->category = MEMBER_UNKNOWN;
  } else if (inGlobal) {
    /* just check the global scope */
    scopes.front()->lookup(o);
  } else {
    for (auto iter = scopes.rbegin(); iter != scopes.rend() && !o->category;
        ++iter) {
      (*iter)->lookup(o);
    }
    if (!o->category) {
      if (currentClass) {
        /* While a local or global should be resolved, a member may not be
         * if it belongs to a base class. This is particularly the case when
         * the "curiously recurring template pattern" (CRTP) is used, where
         * the base class is given by a generic type parameter, and the
         * particular type is unknown at this point (and won't be known,
         * without type deduction). In this case, categorize as
         * MEMBER_UNKNOWN, and let the C++ handle the rest. */
        o->category = MEMBER_UNKNOWN;
      } else {
        throw UnresolvedException(o);
      }
    }
  }
  return o;
}

bi::Type* bi::Resolver::modify(NamedType* o) {
  ScopedModifier::modify(o);
  for (auto iter = scopes.rbegin(); iter != scopes.rend() && !o->category;
      ++iter) {
    (*iter)->lookup(o);
  }
  if (!o->category) {
    throw UnresolvedException(o);
  }
  return o;
}

bi::Statement* bi::Resolver::modify(Class* o) {
  scopes.back()->inherit(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Resolver::modify(Fiber* o) {
  /* resolve start function, using new Resolver for correct scoping */
  Resolver resolver(currentPackage, currentClass);
  o->start = o->start->accept(&resolver);

  return ScopedModifier::modify(o);
}

bi::Statement* bi::Resolver::modify(MemberFiber* o) {
  /* resolve start function, using new Resolver for correct scoping */
  Resolver resolver(currentPackage, currentClass);
  o->start = o->start->accept(&resolver);

  return ScopedModifier::modify(o);
}

bi::Statement* bi::Resolver::modify(Yield* o) {
  if (o->resume) {
    /* resolve resume function, using new Resolver for correct scoping */
    Resolver resolver(currentPackage, currentClass);
    o->resume = o->resume->accept(&resolver);
  }
  return ScopedModifier::modify(o);
}
