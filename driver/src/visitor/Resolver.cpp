/**
 * @file
 */
#include "src/visitor/Resolver.hpp"

#include "src/exception/all.hpp"
#include "src/visitor/all.hpp"
#include "src/build/misc.hpp"

birch::Resolver::Resolver(Package* currentPackage, Class* currentClass,
    Fiber* currentFiber) :
    ScopedModifier(currentPackage, currentClass, currentFiber) {
  //
}

birch::Resolver::~Resolver() {
  //
}

birch::Statement* birch::Resolver::modify(ExpressionStatement* o) {
  ScopedModifier::modify(o);

  /* warn about use of old implicit spin */
  auto call = dynamic_cast<Call*>(o->single);
  if (call) {
    auto named = dynamic_cast<NamedExpression*>(call->single);
    if (named && (named->category == GLOBAL_FIBER ||
        named->category == MEMBER_FIBER)) {
      warn("implicit running of fibers is no longer supported, use postfix !! operator instead, i.e. f(a, b, c)!!.", o->loc);
    }
  }

  return o;
}

birch::Expression* birch::Resolver::modify(Parameter* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

birch::Statement* birch::Resolver::modify(LocalVariable* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

birch::Expression* birch::Resolver::modify(NamedExpression* o) {
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
        throw UndefinedException(o);
      }
    }
  }
  return o;
}

birch::Type* birch::Resolver::modify(NamedType* o) {
  ScopedModifier::modify(o);
  for (auto iter = scopes.rbegin(); iter != scopes.rend() && !o->category;
      ++iter) {
    (*iter)->lookup(o);
  }
  if (!o->category) {
    throw UndefinedException(o);
  }
  return o;
}

birch::Statement* birch::Resolver::modify(Class* o) {
  scopes.back()->inherit(o);
  return ScopedModifier::modify(o);
}

birch::Statement* birch::Resolver::modify(Fiber* o) {
  /* resolve start function, using new Resolver for correct scoping */
  Resolver resolver(currentPackage, currentClass);
  o->start = o->start->accept(&resolver);

  return ScopedModifier::modify(o);
}

birch::Statement* birch::Resolver::modify(MemberFiber* o) {
  /* resolve start function, using new Resolver for correct scoping */
  Resolver resolver(currentPackage, currentClass);
  o->start = o->start->accept(&resolver);

  return ScopedModifier::modify(o);
}

birch::Statement* birch::Resolver::modify(Yield* o) {
  if (o->resume) {
    /* resolve resume function, using new Resolver for correct scoping */
    Resolver resolver(currentPackage, currentClass);
    o->resume = o->resume->accept(&resolver);
  }
  return ScopedModifier::modify(o);
}
